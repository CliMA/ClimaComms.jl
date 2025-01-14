import Logging, LoggingExtras

export MPILogger, MPIFileLogger

"""
    default_logger()
    default_logger(ctx::AbstractCommsContext)

Return the default logger for the given context. If no context is passed, obtain the default context.

Passing an MPICommsContext will return the `MPILogger`.
Any other context will return the default `ConsoleLogger`.
"""
default_logger() = default_logger(context())
default_logger(ctx::MPICommsContext) = MPILogger(ctx)
default_logger(ctx::AbstractCommsContext) = Logging.ConsoleLogger(stdout)

"""
    enable_root_logger(logger)

Only log messages from the root process for the given `logger`.
"""
function enable_root_logger(logger)
    if ClimaComms.iamroot(comms_ctx)
        Logging.global_logger(logger)
    else
        Logging.global_logger(Logging.NullLogger())
    end
end

"""
    MPILogger(context::AbstractCommsContext)
    MPILogger(iostream, context)
    
Add a rank prefix before log messages.

Outputs to `stdout` if no IOStream is given.
"""
MPILogger(ctx::AbstractCommsContext) = MPILogger(stdout, ctx)

function MPILogger(iostream, ctx::AbstractCommsContext)
    pid = mypid(ctx)

    function format_log(io, log)
        print(io, "[P$pid] ")
        println(io, " $(log.level): $(log.message)")
    end

    return LoggingExtras.FormatLogger(format_log, iostream)
end

"""
    MPIFileLogger(context, log_dir)

Log MPI ranks to different files within the `log_dir`.
"""
function MPIFileLogger(
    ctx::AbstractCommsContext,
    log_dir::AbstractString;
    min_level::Logging.LogLevel = Logging.Info,
)
    rank = mypid(ctx)
    !isdir(log_dir) && mkdir(log_dir)
    return LoggingExtras.FileLogger(
        joinpath(log_dir, "rank_$rank.log");
        append = true,
        always_flush = true,
    )
end
