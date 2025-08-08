# Model Parameters

We name all the parameters the same as they appear in various literatures.

## Fault Parameter Space

### RateStateQuasiDynamicProperty
  
| Name   | Type              | Description                       |
|--------|-------------------|-----------------------------------|
| `a`    | Array{<:Real}     | Direct effect parameter           |
| `b`    | Array{<:Real}     | Evolution effect parameter        |
| `L`    | Array{<:Real}     | Critical slip distance            |
| `σ`    | Array{<:Real}     | Effective normal stress           |
| `η`    | Real              | Radiation damping coefficient     |
| `vpl`  | Real              | Plate velocity (loading rate)     |
| `f₀`   | Real              | Reference friction coefficient    |
| `v₀`   | Real              | Reference slip velocity           |

### DilatancyProperty

| Name   | Type              | Description                       |
|--------|-------------------|-----------------------------------|
| `tₚ`   | Array{<:Real}     | Characteristic diffusion timescale|
| `ϵ`    | Array{<:Real}     | Dilatancy coefficient             |
| `β`    | Array{<:Real}     | Fault gouge bulk compressibility  |
| `p₀`   | Array{<:Real}     | Ambient pore pressure             |

## Mantle Parameter Space

### PowerLawViscosityProperty

| Name    | Type              | Description                        |
|---------|-------------------|------------------------------------|
| `γ`     | Array{<:Real}     | Power law coefficient              |
| `n`     | Array{<:Real}     | (Power - 1) of the stress term     |
| `dϵ₀`   | Vector{<:Real}    | Reference strain rate              |

### CompositePowerLawViscosityProperty

| Name     | Type                        | Description                        |
|----------|-----------------------------|------------------------------------|
| `piter`  | Vector{<:ViscosityProperty} | Series of different viscosity laws |
| `dϵ₀`    | Vector{<:Real}              | Reference strain rate              |
