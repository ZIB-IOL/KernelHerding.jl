using KernelHerding
using Documenter

DocMeta.setdocmeta!(KernelHerding, :DocTestSetup, :(using KernelHerding); recursive=true)

makedocs(;
    modules=[KernelHerding],
    authors="Mathieu Besançon <mathieu.besancon@gmail.com> and contributors",
    repo="https://github.com/matbesancon/KernelHerding.jl/blob/{commit}{path}#{line}",
    sitename="KernelHerding.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://matbesancon.github.io/KernelHerding.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/matbesancon/KernelHerding.jl",
    devbranch="main",
)
