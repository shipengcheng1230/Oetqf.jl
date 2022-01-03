using Documenter, Oetqf

makedocs(
    doctest=false,
    modules = [Oetqf],
    sitename = "Oetqf",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "APIs" => "APIs.md",
    ],
)

deploydocs(
  repo = "github.com/shipengcheng1230/Oetqf.jl.git",
  target = "build",
)