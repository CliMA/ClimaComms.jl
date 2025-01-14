# Logging

Julia's standard library, `Logging.jl`, provides logging functionality that ClimaComms builds upon. To return the default logger for the given `CommsContext`, use [`ClimaComms.default_logger(context)`](@ref).

As of v1.11, Julia's default logger is [`Logging.ConsoleLogger()`](https://docs.julialang.org/en/v1/stdlib/Logging/#Base.CoreLogging.ConsoleLogger). The current logger for your Julia session can be determined via [`Logging.current_logger()`](https://docs.julialang.org/en/v1/stdlib/Logging/#Logging.current_logger).

ClimaComms provides two loggers for use with MPI:

- [`MPILogger(context)`](@ref): Adds an MPI rank prefix before all log messages.
- [`MPIFileLogger(context, log_dir)`](@ref): Logs MPI ranks to different files within the `log_dir`.


## Log Levels

Log levels are used to set the severity/verbosity of a log record.

There are several defaults defined by Logging.jl: `Debug`, `Info`, `Warn`, and `Error`.
See [here](https://docs.julialang.org/en/v1/stdlib/Logging/#Log-event-structure) for a full description of the levels.


Custom logging levels can be defined using `Logging.LogLevel(level)`

To disable all log messages at log levels equal to or less than a given `LogLevel`, use [`Logging.disable_logging(level)`](https://docs.julialang.org/en/v1/stdlib/Logging/#Logging.disable_logging).

## Filtering Log Messages

[LoggingExtras.jl](https://github.com/JuliaLogging/LoggingExtras.jl) provides the `EarlyFilteredLogger(filter, logger)`, which takes two arguments:

- `filter(log_args)` is a function which takes in `log_args` and returns a Boolean determining if the message should be logged. `log_args` is a NamedTuple with fields `level`, `_module`, `id` and `group`. Example `filter` functions are provided below in the Common Use Cases.
- `logger` is any existing logger, such as `Logging.ConsoleLogger()` or `MPILogger(ctx)`.

### Common Use Cases

#### How do I filter out warning messages?

```julia
using Logging, LoggingExtras

function no_warnings(log_args)
    return log_args.level != Logging.Warn
end

filtered_logger = EarlyFilteredLogger(no_warnings, Logging.current_logger())

with_logger(filtered_logger) do
    @warn "Hide this warning"
    @info "Display this message"
end
```

#### How do I filter out messages from certain modules?

```julia
using Logging, LoggingExtras

function module_filter(excluded_modules)
    return function(log_args)
        !(log_args._module in excluded_modules)
    end
end

ModuleFilteredLogger(excluded) =
    EarlyFilteredLogger(module_filter(excluded), Logging.current_logger())
# To test this logger:
module TestModule
    using Logging
    function log_something()
        @info "This message will appear"
    end
end

excluded = (Main, Base)
with_logger(ModuleFilteredLogger(excluded)) do
    @info "Hide this message"
    TestModule.log_something()
end
```

