import Logging, LoggingExtras

export MPILogger, FileLogger

"""
    OnlyRootLogger()
    OnlyRootLogger(ctx::AbstractCommsContext)

Return a logger that prints to the console on the root process and
silences all other processes.

If no context is passed, obtain the default context via [`context`](@ref).
For MPI runs, this logger is installed as the global logger by the first
call to [`init`](@ref).
"""
OnlyRootLogger() = OnlyRootLogger(context())

function OnlyRootLogger(ctx::AbstractCommsContext)
    if iamroot(ctx)
        return Logging.ConsoleLogger()
    else
        return Logging.NullLogger()
    end
end

"""
    MPILogger(ctx::AbstractCommsContext)
    MPILogger(iostream, ctx::AbstractCommsContext)

Return a logger that prefixes each log message with the process ID (e.g.,
`[P1]`). Output goes to `stdout` if no `iostream` is given.

# Examples
```julia
using Logging
logger = ClimaComms.MPILogger(ClimaComms.context())
global_logger(logger)
@info "Hello"   # prints "[P1]  Info: Hello" on the root process
```
"""
MPILogger(ctx::AbstractCommsContext) = MPILogger(stdout, ctx)

function MPILogger(iostream, ctx::AbstractCommsContext)
    pid = mypid(ctx)

    function format_log(io, log)
        print(io, "[P$pid] ")
        println(io, " $(log.level): $(log.message)")
    end

    logger = LoggingExtras.FormatLogger(format_log, iostream)
    # Wrap in a MinLevelLogger to ensure a minimum log level is discoverable (issue #118)
    return LoggingExtras.MinLevelLogger(logger, Logging.Info)
end

"""
    FileLogger(ctx, log_dir; log_stdout = true, min_level = Logging.Info)

Return a logger that writes each process's log messages to a separate
file in `log_dir` (`rank_1.log`, `rank_2.log`, ...), with
`log_dir/output.log` a symbolic link to the root process's file. For
single-process runs, all messages go directly to `log_dir/output.log`.

# Keyword Arguments
- `log_stdout = true`: if `true`, the root process also logs to `stdout`.
- `min_level = Logging.Info`: the minimum level a message must have to be
  logged.

# Examples
```julia
using Logging
logger = ClimaComms.FileLogger(ClimaComms.context(), "logs")
with_logger(logger) do
    @info "Written to logs/output.log and stdout"
end
```
"""
function FileLogger(
    ctx::AbstractCommsContext,
    log_dir;
    log_stdout = true,
    min_level::Logging.LogLevel = Logging.Info,
)
    return FileLogger(stdout, ctx, log_dir; log_stdout, min_level)
end

function FileLogger(
    io::IO,
    ctx::MPICommsContext,
    log_dir::AbstractString;
    log_stdout = true,
    min_level::Logging.LogLevel = Logging.Info,
)
    mpi_log_dir = joinpath(log_dir, "logs")
    !isdir(mpi_log_dir) && mkpath(mpi_log_dir)
    ClimaComms.barrier(ctx)  # Ensure that the folder is created
    rank = mypid(ctx)
    filepath = abspath(joinpath(mpi_log_dir, "rank_$rank.log"))

    # Link output.log to the root process's log file, replacing a stale
    # link left behind by a previous run in the same directory. If a
    # regular file with that name exists, leave it untouched.
    if iamroot(ctx)
        symlink_path = abspath(joinpath(log_dir, "output.log"))
        islink(symlink_path) && rm(symlink_path)
        ispath(symlink_path) || symlink(filepath, symlink_path)
    end

    function min_level_filter(log_args)
        return log_args.level >= min_level
    end

    file_logger = LoggingExtras.FormatLogger(
        format_log,
        filepath,
        append = true,
        always_flush = true,
    )

    filtered_logger =
        LoggingExtras.EarlyFilteredLogger(min_level_filter, file_logger)

    logger = if iamroot(ctx) && log_stdout
        LoggingExtras.TeeLogger((Logging.ConsoleLogger(io), filtered_logger))
    else
        filtered_logger
    end
    # Wrap in a MinLevelLogger to ensure a minimum log level is discoverable (issue #118)
    return LoggingExtras.MinLevelLogger(logger, min_level)
end

function FileLogger(
    io::IO,
    ctx::SingletonCommsContext,
    log_dir::AbstractString;
    log_stdout = true,
    min_level::Logging.LogLevel = Logging.Info,
)
    !isdir(log_dir) && mkpath(log_dir)
    filepath = joinpath(log_dir, "output.log")

    function min_level_filter(log_args)
        return log_args.level >= min_level
    end

    file_logger = LoggingExtras.FormatLogger(
        format_log,
        filepath,
        append = true,
        always_flush = true,
    )

    filtered_logger =
        LoggingExtras.EarlyFilteredLogger(min_level_filter, file_logger)

    logger = if log_stdout
        LoggingExtras.TeeLogger((Logging.ConsoleLogger(io), filtered_logger))
    else
        filtered_logger
    end
    # Wrap in a MinLevelLogger to ensure a minimum log level is discoverable (issue #118)
    return LoggingExtras.MinLevelLogger(logger, min_level)
end

"""
    format_log(io, args)

Format log messages similarly to `Logging.ConsoleLogger`, with box
decorations for multiline messages, indentation, and bolding.

Called from [`FileLogger`](@ref).
"""
function format_log(io::IO, args)
    msg = string(args.message)
    msglines = split(msg, '\n')
    if !isempty(args.kwargs)
        for (key, val) in args.kwargs
            push!(msglines, "$key = $val")
        end
    end

    level_prefix = string(args.level)

    for (i, msg) in enumerate(msglines)
        # Set up the box decoration for multi-line strings
        boxstr =
            length(msglines) == 1 ? "[ " :
            i == 1 ? "┌ " : i < length(msglines) ? "│ " : "└ "
        printstyled(io, boxstr, bold = true)
        if i == 1
            printstyled(io, level_prefix, ": ", bold = true)
        end
        indent = i == 1 ? 0 : 2
        print(io, " "^indent, msg)
        println(io)
    end
end

"""
    with_tempdir(f::Function, ctx::AbstractCommsContext)

Create a temporary directory on the root process, broadcast its path to
all processes, and call `f` on the path. All processes receive the same
path, so the directory can be used for files shared across the run.
"""
function with_tempdir(f::Function, ctx)
    temp_dir = ClimaComms.iamroot(ctx) ? mktempdir() : nothing
    temp_dir = ClimaComms.bcast(ctx, temp_dir)
    return f(temp_dir)
end
