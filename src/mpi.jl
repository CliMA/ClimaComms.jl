import Logging

"""
    MPICommsContext()
    MPICommsContext(device)
    MPICommsContext(device, comm)

A MPI communications context, used for distributed runs.
[`AbstractCPUDevice`](@ref) and [`CUDADevice`](@ref) device options are currently supported.
"""
struct MPICommsContext{D <: AbstractDevice, C} <: AbstractCommsContext
    device::D
    mpicomm::C
end

function MPICommsContext end

# Adapted from Logging.ConsoleLogger
# https://docs.julialang.org/en/v1/stdlib/Logging/#AbstractLogger-interface
# https://github.com/JuliaLang/julia/blob/5e9a32e7af2837e677e60543d4a15faa8d3a7297/base/logging/ConsoleLogger.jl
"""
    MPILogger()


"""
struct MPILogger <: Logging.AbstractLogger
    stream::IO
    min_level::Logging.LogLevel
    meta_formatter::Any
    show_limited::Bool # TODO: Use this to limit stracktraces
    ctx::ClimaComms.MPICommsContext
end

function MPILogger end

Logging.shouldlog(logger::MPILogger, level, _module, group, id) =
    get(logger.message_limits, id, 1) > 0

Logging.min_enabled_level(logger::MPILogger) = logger.min_level

# TODO: Add comments explaining this complexity, copied from ConsoleLogger
function Logging.handle_message(
    logger::MPILogger,
    level,
    message,
    _module,
    group,
    id,
    file,
    line,
)
    @nospecialize
    # Generate a text representation of the message and all key value pairs,
    # split into lines.  This is specialised to improve type inference,
    # and reduce the risk of resulting method invalidations.
    message = string(message)
    msglines =
        if Base._isannotated(message) && !isempty(Base.annotations(message))
            message =
                Base.AnnotatedString(String(message), Base.annotations(message))
            @NamedTuple{
                    indent::Int,
                    msg::Union{
                        SubString{Base.AnnotatedString{String}},
                        SubString{String},
                    },
                }[
                (indent = 0, msg = l) for l in split(chomp(message), '\n')
            ]
        else
            [
                (indent = 0, msg = l) for
                l in split(chomp(convert(String, message)::String), '\n')
            ]
        end

    stream = logger.stream
    color, prefix, suffix = logger.meta_formatter(
        level,
        _module,
        group,
        id,
        filepath,
        line,
    )::Tuple{Union{Symbol, Int}, String, String}
    minsuffixpad = 2
    buf = IOBuffer()
    iob = IOContext(buf, stream)
    nonpadwidth =
        2 +
        (isempty(prefix) || length(msglines) > 1 ? 0 : length(prefix) + 1) +
        msglines[end].indent +
        termlength(msglines[end].msg) +
        (isempty(suffix) ? 0 : length(suffix) + minsuffixpad)
    justify_width = min(logger.right_justify, dsize[2])
    if nonpadwidth > justify_width && !isempty(suffix)
        push!(msglines, (indent = 0, msg = SubString("")))
        minsuffixpad = 0
        nonpadwidth = 2 + length(suffix)
    end

    for (i, (indent, msg)) in enumerate(msglines)
        boxstr =
            length(msglines) == 1 ? "[ " :
            i == 1 ? "┌ " : i < length(msglines) ? "│ " : "└ "
        printstyled(iob, boxstr, bold = true, color = color)
        if i == 1 && !isempty(prefix)
            printstyled(iob, prefix, " ", bold = true, color = color)
        end
        print(iob, " "^indent, msg)
        if i == length(msglines) && !isempty(suffix)
            npad = max(0, justify_width - nonpadwidth) + minsuffixpad
            print(iob, " "^npad)
            printstyled(iob, suffix, color = :light_black)
        end
        println(iob)
    end

    write(stream, take!(buf))
    nothing
end

function mpi_metafmt(level::Logging.LogLevel, _module, group, id, file, line)
    @nospecialize
    color = default_logcolor(level)
    prefix = string(
        "[R$(ClimaComms.mypid(ClimaComms.ctx()))] ",
        level == Warn ? "Warning" : string(level),
        ':',
    )
    suffix::String = ""
    Info <= level < Warn && return color, prefix, suffix
    _module !== nothing && (suffix *= string(_module)::String)
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= file::String
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end

# function Logging.catch_exceptions(logger::MPILogger) end

# Sample utility function
function MPIFileLogger(base_path::String)
    function stream_generator(rank)
        filename = "$(base_path)_$(rank)"
        open(filename, "a")  # Append mode
    end
    return MPILogger(stream_generator)
end
