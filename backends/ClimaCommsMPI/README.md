## ClimaCommsMPI

MPI backend for ClimaComms.

### Use ClimaCommsMPI from the ClimaComms parent package

    cd ClimaComms
    julia --project

Activate the Pkg repl mode and `dev` the ClimaCommsMPI subpackage:

    (ClimaCore) pkg> dev backends/ClimaCommsMPI

You can now use ClimaCommsMPI in your ClimaComms pkg environment:

    julia> using ClimaCommsMPI
    julia> ClimaComms.init(MPICommsContext)

### Development of the `ClimaCommsMPI` subpackage

    cd ClimaComms/backends/ClimaCommsMPI

    # Add ClimaComms to subpackage environment
    julia --project -e 'using Pkg; Pkg.develop("../../")

    # Instantiate ClimaCommsMPI project environment
    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -e 'using Pkg; Pkg.test()'
