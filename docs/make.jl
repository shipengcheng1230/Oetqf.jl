using Documenter, Oetqf

using Literate

EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples")
OUTPUT = joinpath(@__DIR__, "src/generated")

for example ∈ filter!(x -> endswith(x, ".jl"), readdir(EXAMPLE_DIR))
    Literate.markdown(abspath(joinpath(EXAMPLE_DIR, example)), OUTPUT)
end

makedocs(
    doctest=false,
    modules = [Oetqf],
    sitename = "Oetqf",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "A 2D transform fault overlaying a 3D mantle" => "generated/otf-with-mantle.md",
        ],
        "Parameters" => "parameters.md",
        "Testing" => "testing.md",
        "APIs" => "APIs.md",
    ],
)

deploydocs(
  repo = "github.com/shipengcheng1230/Oetqf.jl.git",
  target = "build",
)