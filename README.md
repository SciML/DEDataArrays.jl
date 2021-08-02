# DEDataArrays.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/DEDataArrays.jl/workflows/CI/badge.svg)](https://github.com/SciML/DEDataArrays.jl/actions?query=workflow%3ACI)

The `DEDataArray{T}` type allows one to add other "non-continuous" variables
to an array, which can be useful in many modeling situations involving lots of
events. To define an `DEDataArray`, make a type which subtypes `DEDataArray{T}`
with a field `x` for the "array of continuous variables" for which you would
like the differential equation to treat directly. The other fields are treated
as "discrete variables". For example:

```julia
mutable struct MyDataArray{T,1} <: DEDataArray{T,1}
    x::Array{T,1}
    a::T
    b::Symbol
end
```

In this example, our resultant array is a `SimType`, and its data which is presented
to the differential equation solver will be the array `x`. Any array which the
differential equation solver can use is allowed to be made as the field `x`, including
other `DEDataArray`s. Other than that, you can add whatever fields you please, and
let them be whatever type you please.

These extra fields are carried along in the differential equation solver that
the user can use in their `f` equation and modify via callbacks. For example,
inside of a an update function, it is safe to do:

```julia
function f(du,u,p,t)
  u.a = t
end
```

to update the discrete variables (unless the algorithm notes that it does not
step to the endpoint, in which case a callback must be used to update appropriately.)

Note that the aliases `DEDataVector` and `DEDataMatrix` cover the one and two
dimensional cases.

### Example: A Control Problem

In this example we will use a `DEDataArray` to solve a problem where control parameters
change at various timepoints. First we will build

```julia
mutable struct SimType{T} <: DEDataVector{T}
    x::Array{T,1}
    f1::T
end
```

as our `DEDataVector`. It has an extra field `f1` which we will use as our control
variable. Our ODE function will use this field as follows:

```julia
function f(du,u,p,t)
    du[1] = -0.5*u[1] + u.f1
    du[2] = -0.5*u[2]
end
```

Now we will setup our control mechanism. It will be a simple setup which uses
set timepoints at which we will change `f1`. At `t=5.0` we will want to increase
the value of `f1`, and at `t=8.0` we will want to decrease the value of `f1`. Using
the [`DiscreteCallback` interface](@ref discrete_callback), we code these conditions
as follows:

```julia
const tstop1 = [5.]
const tstop2 = [8.]


function condition(u,t,integrator)
  t in tstop1
end

function condition2(u,t,integrator)
  t in tstop2
end
```

Now we have to apply an effect when these conditions are reached. When `condition`
is hit (at `t=5.0`), we will increase `f1` to 1.5. When `condition2` is reached,
we will decrease `f1` to `-1.5`. This is done via the functions:

```julia
function affect!(integrator)
  for c in full_cache(integrator)
    c.f1 = 1.5
  end
end

function affect2!(integrator)
  for c in full_cache(integrator)
    c.f1 = -1.5
  end
end
```

Notice that we have to loop through the `full_cache` array (provided by the integrator
interface) to ensure that all internal caches are also updated. With these functions
we can build our callbacks:

```julia
save_positions = (true,true)

cb = DiscreteCallback(condition, affect!, save_positions=save_positions)

save_positions = (false,true)

cb2 = DiscreteCallback(condition2, affect2!, save_positions=save_positions)

cbs = CallbackSet(cb,cb2)
```


Now we define our initial condition. We will start at `[10.0;10.0]` with `f1=0.0`.

```julia
u0 = SimType([10.0;10.0], 0.0)
prob = ODEProblem(f,u0,(0.0,10.0))
```

Lastly we solve the problem. Note that we must pass `tstop` values of `5.0` and
`8.0` to ensure the solver hits those timepoints exactly:

```julia
const tstop = [5.;8.]
sol = solve(prob,Tsit5(),callback = cbs, tstops=tstop)
```

![data_array_plot](https://user-images.githubusercontent.com/1814174/127798873-624f3f37-e89b-4938-8088-b51107d278a1.png)

It's clear from the plot how the controls affected the outcome.

### Data Arrays vs Parameterized Functions

The reason for using a `DEDataArray` is because the solution will then save the
control parameters. For example, we can see what the control parameter was at
every timepoint by checking:

```julia
[sol[i].f1 for i in 1:length(sol)]
```

A similar solution can be achieved using a `ParameterizedFunction`.
We could have instead created our function as:

```julia
function f(du,u,p,t)
    du[1] = -0.5*u[1] + p
    du[2] = -0.5*u[2]
end
u0 = SimType([10.0;10.0], 0.0)
p = 0.0
prob = ODEProblem(f,u0,(0.0,10.0),p)
const tstop = [5.;8.]
sol = solve(prob,Tsit5(),callback = cbs, tstops=tstop)
```

where we now change the callbacks to changing the parameter:

```julia
function affect!(integrator)
  integrator.p = 1.5
end

function affect2!(integrator)
  integrator.p = -1.5
end
```

This will also solve the equation and get a similar result. It will also be slightly
faster in some cases. However, if the equation is solved in this manner, there will
be no record of what the parameter was at each timepoint. That is the tradeoff
between `DEDataArray`s and `ParameterizedFunction`s.

## Downsides of DEDataArrays

DEDataArray is not a good idea. [This explains why](https://discourse.julialang.org/t/diffeqs-hybrid-continuous-discrete-system-periodic-callback/23791/19?u=chrisrackauckas). But, this repo will stay alive to keep it around for
people who still want to use it.
