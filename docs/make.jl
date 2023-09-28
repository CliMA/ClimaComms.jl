using Documenter, ClimaComms

format = Documenter.HTML(
    prettyurls = !isempty(get(ENV, "CI", "")),
    collapselevel = 1,
)
makedocs(
    sitename = "ClimaComms.jl",
    warnonly = true,
    format = format,
    checkdocs = :exports,
    clean = true,
    doctest = true,
    modules = [ClimaComms],
    pages = Any["Home" => "index.md"],
)
deploydocs(
    repo = "github.com/CliMA/ClimaComms.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
