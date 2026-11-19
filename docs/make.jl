if joinpath(@__DIR__, "..") ∉ LOAD_PATH
    push!(LOAD_PATH, joinpath(@__DIR__, ".."))
end
using Documenter, ClimaComms

format = Documenter.HTML(
    prettyurls = !isempty(get(ENV, "CI", "")),
    collapselevel = 1,
)
makedocs(
    sitename = "ClimaComms.jl",
    strict = false,
    format = format,
    checkdocs = :exports,
    clean = true,
    doctest = true,
    modules = [ClimaComms],
    pages = Any["Home" => "index.md", "API" => "api.md"],
)
deploydocs(
    repo = "github.com/CliMA/ClimaComms.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
