# Note: This must be the first file executed in the tests.
#
# The following structure is used to test the problem reported at issue #507. We
# need to define it here because, since `recursivecopy!` and `copy_fields!` are
# generated functions, then the `ArrayInterface.isimmutable` method must be
# defined before the first `using DiffEqBase` in a Julia session.

struct Quaternion{T} <: AbstractVector{T}
    q0::T
    q1::T
    q2::T
    q3::T
end

using ArrayInterface
ArrayInterface.ismutable(::Type{<:Quaternion}) = false
Base.size(::Quaternion) = 4

# https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/525

using DEDataArrays, OrdinaryDiffEq, StochasticDiffEq
using StaticArrays, RecursiveArrayTools, LinearAlgebra, Test

mutable struct VectorType{T} <: DEDataVector{T}
    x::Vector{T}
    f::T
end

mutable struct MatrixType{T, S} <: DEDataMatrix{T}
    f::S
    x::Matrix{T}
end

mutable struct SimType2{T} <: DEDataVector{T}
    x::Vector{T}
    y::Vector{T}
    u::Vector{T}
end

A = [0.0; 1.0]
B = [1.0 2; 4 3]

a = VectorType{Float64}(copy(A), 1.0)
b = MatrixType{Float64, Float64}(2.0, copy(B))

# basic methods of AbstractArray interface
@test eltype(a) == Float64 && eltype(b) == Float64

# size
@test length(a) == 2 && length(b) == 4
@test size(a) == (2,) && size(b) == (2, 2)

# iteration
@test first(a) == 0.0 && first(b) == 1
@test last(a) == 1.0 && last(b) == 3
for (i, (ai, Ai)) in enumerate(zip(a, A))
    @test ai == Ai
end
for (i, (bi, Bi)) in enumerate(zip(b, B))
    @test bi == Bi
end

# indexing
@test eachindex(a) == Base.LinearIndices(A) && eachindex(b) == vec(Base.LinearIndices(B))
@test a[2] == a[end] == 1.0
@test b[3] == b[1, 2] == 2 && b[:, 1] == [1; 4]

a[1] = 3;
b[2, 1] = 1;
@test a[1] == 3.0 && b[2] == 1

a[:] = A;
b[:, 1] = B[1:2];
@test a.x == A && b.x == B

# simple broadcasts
@test [1 // 2] .* a == [0.0; 0.5]
@test b ./ 2 == [0.5 1.0; 2.0 1.5]
@test b .+ a == [1.0 2.0; 5.0 4.0]
@test_broken a .+ b == [1.0 2.0; 5.0 4.0] # Doesn't find the largest

# similar data arrays
a2 = similar(a);
b2 = similar(b, (1, 4));
b3 = similar(b, Float64, (1, 4));
@test typeof(a2.x) == Vector{Float64} && a2.f == 1.0
@test typeof(b2.x) == Matrix{Float64} && b2.f == 2.0
@test typeof(b3.x) == Matrix{Float64} && b3.f == 2.0

# copy all fields of data arrays
recursivecopy!(a2, a)
recursivecopy!(b2, b)
@test a2.x == A && vec(b2.x) == vec(B)
recursivecopy!(b3, b)

# copy all fields except of field x
a2.f = 3.0;
b2.f = -1;
b3.f == 0;
DEDataArrays.copy_fields!(a, a2)
DEDataArrays.copy_fields!(b, b2)
@test a.f == 3.0 && b.f == -1.0
DEDataArrays.copy_fields!(b, b3)

# create data array with field x replaced by new array
a3 = DEDataArrays.copy_fields([1.0; 0.0], a)
@test a3 == VectorType{Float64}([1; 0], 3.0)

# broadcast assignments
a.f = 0.0
a .= a2 .+ a3
@test a == VectorType{Float64}([1.0; 1.0], 0.0)
@test a == a2 .+ a3
@test (a2 .+ a3) isa VectorType

old_b = copy(b)
b .= b .^ 2 .+ a3
@test b == MatrixType{Int, Float64}(-1.0, [2 5; 16 9])
@test b == old_b .^ 2 .+ a3
@test (b .^ 2 .+ a3) isa MatrixType

# Test ability to use MVectors in the solvers
mutable struct SimWorkspace{T} <: DEDataVector{T}
    x::MVector{2, T}
    a::T
end
s0 = SimWorkspace{Float64}(MVector{2, Float64}(1.0, 4.0), 1.0)
similar(s0, Float64, size(s0))
s1 = SimWorkspace{Float64}(MVector{2, Float64}(2.0, 1.0), 1.0)
s0 .+ s1 == SimWorkspace{Float64}(MVector{2, Float64}(3.0, 5.0), 1.0)

mutable struct SimWorkspace2{T} <: DEDataVector{T}
    x::SVector{2, T}
    a::T
end
s0 = SimWorkspace2{Float64}(SVector{2, Float64}(1.0, 4.0), 1.0)
s1 = SimWorkspace2{Float64}(SVector{2, Float64}(2.0, 1.0), 1.0)
s0 .+ s1 == SimWorkspace2{Float64}(SVector{2, Float64}(3.0, 5.0), 1.0)

# Test `recursivecopy!` in immutable structures derived from `AbstractArrays`.
# See issue #507.
mutable struct SimWorkspace3{T} <: DEDataVector{T}
    x::Vector{T}
    q::Quaternion{T}
end

a = SimWorkspace3([1.0, 2.0, 3.0], Quaternion(cosd(15), 0.0, 0.0, sind(15)))
b = SimWorkspace3([0.0, 0.0, 0.0], Quaternion(1.0, 0.0, 0.0, 0.0))

recursivecopy!(b, a)
@test b.x == a.x
@test b.q.q0 == a.q.q0
@test b.q.q1 == a.q.q1
@test b.q.q2 == a.q.q2
@test b.q.q3 == a.q.q3

a = SimWorkspace3([1.0, 2.0, 3.0], Quaternion(cosd(15), 0.0, 0.0, sind(15)))
b = SimWorkspace3([0.0, 0.0, 0.0], Quaternion(1.0, 0.0, 0.0, 0.0))

DEDataArrays.copy_fields!(b, a)
@test b.x == [0.0, 0.0, 0.0]
@test b.q.q0 == a.q.q0
@test b.q.q1 == a.q.q1
@test b.q.q2 == a.q.q2
@test b.q.q3 == a.q.q3

mutable struct SimType{T} <: DEDataVector{T}
    x::Array{T, 1}
    f1::T
end

function f(u, p, t) # new out-of-place definition
    SimType([-0.5 * u[1] + u.f1,
                -0.5 * u[2]], u.f1)
end

function f!(du, u, p, t) # old in-place definition
    du[1] = -0.5 * u[1] + u.f1
    du[2] = -0.5 * u[2]
end

tstop1 = [5.0]
tstop2 = [8.0]

function condition(u, t, integrator)
    t in tstop1
end

function condition2(u, t, integrator)
    t in tstop2
end

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

function affect!_oop(integrator)
    integrator.u.f1 = 1.5
end

function affect2!_oop(integrator)
    integrator.u.f1 = 1.5
end

save_positions = (true, true)
cb = DiscreteCallback(condition, affect!, save_positions = save_positions)
save_positions = (false, true)
cb2 = DiscreteCallback(condition2, affect2!, save_positions = save_positions)
cbs = CallbackSet(cb, cb2)

cb_oop = DiscreteCallback(condition, affect!_oop, save_positions = save_positions)
save_positions = (false, true)
cb2_oop = DiscreteCallback(condition2, affect2!_oop, save_positions = save_positions)
cbs_oop = CallbackSet(cb_oop, cb2_oop)

u0 = SimType([10.0; 10.0], 0.0)

prob_inplace = ODEProblem(f!, u0, (0.0, 10.0))
prob = ODEProblem(f, u0, (0.0, 10.0))

tstop = [5.0; 8.0]

sol = solve(prob_inplace, Tsit5(), callback = cbs, tstops = tstop)
sol = solve(prob, Tsit5(), callback = cbs_oop, tstops = tstop)

# https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/336

AA = SMatrix{2, 2}([0 1;
                    0 0])

mutable struct MyStruct{T} <: DEDataVector{T}
    x::MVector{2, T}
    a::SVector{2, T}
end

function dyn(du, u, t, p)
    u.a = SVector{2}(0.0, 0.1)
    du .= AA * u.x + u.a

    return nothing
end

u0 = MyStruct(MVector{2}(0.0, 0.0), SVector{2}(0.0, 0.0))
prob = ODEProblem(dyn, u0, (0.0, 10.0))

@test copy(u0) isa MyStruct
@test zero(u0) isa MyStruct
@test similar(u0) isa MyStruct
@test similar(u0, Float64) isa MyStruct
@test similar(u0, Float64, size(u0)) isa MyStruct

sol = solve(prob, Tsit5())

# https://github.com/JuliaDiffEq/StochasticDiffEq.jl/issues/247

function f(du, u, p, t)
    du[1] = -0.5 * u[1] + u.f1
    du[2] = -0.5 * u[2]
end

tstop1 = [5.0]
tstop2 = [8.0]

function condition(u, t, integrator)
    t in tstop1
end

function condition2(u, t, integrator)
    t in tstop2
end

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

save_positions = (true, true)

cb = DiscreteCallback(condition, affect!, save_positions = save_positions)

save_positions = (false, true)

cb2 = DiscreteCallback(condition2, affect2!, save_positions = save_positions)

cbs = CallbackSet(cb, cb2)

u0 = SimType([10.0; 10.0], 0.0)
prob = ODEProblem(f, u0, (0.0, 10.0))

tstop = [5.0; 8.0]

# here the new part

function g(du, u, p, t)
    du[1] = 1.0
    du[2] = 1.2
end

dt = 1 / 2^4

prob2 = SDEProblem(f, g, u0, (0.0, 10.0))

# this creates an error

sol = solve(prob2, callback = cbs, tstops = tstop, EM(), dt = dt,
            saveat = collect(8:0.1:10))

# https://github.com/JuliaDiffEq/DiffEqBase.jl/issues/327

struct SimulationState{T, U} <: DEDataArrays.DEDataArray{T, 1}
    x::Vector{T}
    u::U
end

#function Base.convert(::Type{SimulationState{T, U}}, a::AbstractArray{T}) where {T,U}
#    SimulationState(a, zero(eltype(a)))
#end

function open_loop(state, parameters, time)
    v = state[1] * -0.1 + state.u
    return SimulationState([v], state.u)
end

initial_conditions = SimulationState([10.0], 0.0)
time_span = (0.0, 20.0)
ode_prob = ODEProblem(open_loop, initial_conditions, time_span, nothing)

sol = solve(ode_prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

function OrdinaryDiffEq.perform_step!(integrator, cache::OrdinaryDiffEq.FunctionMapCache,
                                      repeat_step = false)
    OrdinaryDiffEq.@unpack u, uprev, dt, t, f, p = integrator
    alg = OrdinaryDiffEq.unwrap_alg(integrator, nothing)
    OrdinaryDiffEq.@unpack tmp = cache
    if integrator.f != OrdinaryDiffEq.DiffEqBase.DISCRETE_INPLACE_DEFAULT &&
       !(typeof(integrator.f) <: OrdinaryDiffEq.DiffEqBase.EvalFunc &&
         integrator.f.f === OrdinaryDiffEq.DiffEqBase.DISCRETE_INPLACE_DEFAULT)
        if OrdinaryDiffEq.FunctionMap_scale_by_time(alg)
            f(tmp, uprev, p, t + dt)
            OrdinaryDiffEq.@muladd OrdinaryDiffEq.@.. broadcast=false u=uprev + dt * tmp
        else
            f(u, uprev, p, t + dt)
        end
        integrator.destats.nf += 1
        if typeof(u) <: DEDataArrays.DEDataArray # Needs to get the fields, since updated uprev
            DEDataArrays.copy_fields!(u, uprev)
        end
    end
end

@testset "DEDataVector" begin
    f = function (du, u, p, t)
        du[1] = -0.5 * u[1] + u.f1
        du[2] = -0.5 * u[2]
    end

    tstop1 = [5.0]
    tstop2 = [8.0]
    tstop = [5.0; 8.0]

    condition = function (u, t, integrator)
        t in tstop1
    end

    affect! = function (integrator)
        for c in OrdinaryDiffEq.full_cache(integrator)
            c.f1 = 1.5
        end
    end

    save_positions = (true, true)

    cb = DiscreteCallback(condition, affect!; save_positions = save_positions)

    condition2 = function (u, t, integrator)
        t in tstop2
    end

    affect2! = function (integrator)
        for c in OrdinaryDiffEq.full_cache(integrator)
            c.f1 = -1.5
        end
    end

    save_positions = (true, true)

    cb2 = DiscreteCallback(condition2, affect2!, save_positions = save_positions)

    cbs = CallbackSet(cb, cb2)

    u0 = SimType{Float64}([10; 10], 0.0)
    prob = ODEProblem(f, u0, (0.0, 10.0))
    sol = solve(prob, Tsit5(), callback = cbs, tstops = tstop)

    sol(1.5:0.5:2.5)

    @test [sol[i].f1 for i in eachindex(sol)] ==
          [fill(0.0, 9); 1.5 * ones(5); -1.5 * ones(4)]

    A = Matrix(Diagonal([0.3, 0.6, 0.9]))
    B = transpose([1 2 3])
    C = [1 / 3 1 / 3 1 / 3]

    function mysystem(t, x, dx, p, u)
        ucalc = u(x, p, t)
        x.u = ucalc
        x.y = C * x.x
        dx .= A * x.x + B * x.u
    end

    input = (x, p, t) -> (1 * one(t) ≤ t ≤ 2 * one(t) ? [one(t)] : [zero(t)])
    prob = DiscreteProblem((dx, x, p, t) -> mysystem(t, x, dx, p, input),
                           SimType2(fill(0.0, 3), fill(0.0, 1), fill(0.0, 1)),
                           (0 // 1, 4 // 1))

    sln = solve(prob, FunctionMap(scale_by_time = false), dt = 1 // 10)

    u1 = [sln[idx].u for idx in 1:length(sln)]
    u2 = [sln(t).u for t in range(0, stop = 4, length = 41)]
    @test any(x -> x[1] > 0, u1)
    @test any(x -> x[1] > 0, u2)

    sln = solve(prob, FunctionMap(scale_by_time = true), dt = 1 // 10)

    u1 = [sln[idx].u for idx in 1:length(sln)]
    u2 = [sln(t).u for t in range(0, stop = 4, length = 41)]
    @test any(x -> x[1] > 0, u1)
    @test any(x -> x[1] > 0, u2)

    sln = solve(prob, Euler(), dt = 1 // 10)

    @test u1 == [sln[idx].u for idx in 1:length(sln)] # Show that discrete is the same
    u1 = [sln[idx].u for idx in 1:length(sln)]
    u2 = [sln(t).u for t in range(0, stop = 4, length = 41)]
    @test any(x -> x[1] > 0, u1)
    @test any(x -> x[1] > 0, u2)

    sln = solve(prob, DP5(), dt = 1 // 10, adaptive = false)

    u1 = [sln[idx].u for idx in 1:length(sln)]
    u2 = [sln(t).u for t in range(0, stop = 4, length = 41)]
    @test any(x -> x[1] > 0, u1)
    @test any(x -> x[1] > 0, u2)
end

######################
# DEDataMatrix
mutable struct SimTypeg{T, T2, N} <: DEDataArray{T, N}
    x::Array{T, N} # two dimensional
    f1::T2
end

@testset "DEDataMatrix" begin
    tstop1 = [10.0]
    tstop2 = [300.0]

    function mat_condition(u, t, integrator)
        t in tstop1
    end

    function mat_condition2(u, t, integrator)
        t in tstop2
    end

    function mat_affect!(integrator)
        for c in OrdinaryDiffEq.full_cache(integrator)
            c.f1 = +1.0
        end
        #  integrator.u[1,1] = 0.001
    end

    function mat_affect2!(integrator)
        for c in OrdinaryDiffEq.full_cache(integrator)
            c.f1 = 0.0
        end
    end

    save_positions = (true, true)
    cb = DiscreteCallback(mat_condition, mat_affect!, save_positions = save_positions)
    save_positions = (false, true)
    cb2 = DiscreteCallback(mat_condition2, mat_affect2!, save_positions = save_positions)
    cbs = CallbackSet(cb, cb2)

    function sigmoid(du, u, p, t)
        du[1, 1] = 0.01 * u[1, 1] * (1 - u[1, 1] / 20)
        du[1, 2] = 0.01 * u[1, 2] * (1 - u[1, 2] / 20)
        du[2, 1] = 0.01 * u[2, 1] * (1 - u[2, 1] / 20)
        du[2, 2] = u.f1 * du[1, 1]
    end

    u0 = SimTypeg(fill(0.00001, 2, 2), 0.0)
    tspan = (0.0, 3000.0)
    prob = ODEProblem(sigmoid, u0, tspan)

    tstop = [tstop1; tstop2]
    sol = solve(prob, Tsit5(), callback = cbs, tstops = tstop)
    sol = solve(prob, Rodas4(), callback = cbs, tstops = tstop)
    sol = solve(prob, Kvaerno3(), callback = cbs, tstops = tstop)
    sol = solve(prob, Rodas4(autodiff = false), callback = cbs, tstops = tstop)
    sol = solve(prob, Kvaerno3(autodiff = false), callback = cbs, tstops = tstop)
end

mutable struct CtrlSimTypeScalar{T, T2} <: DEDataVector{T}
    x::Vector{T}
    ctrl_x::T2 # controller state
    ctrl_u::T2 # controller output
end

@testset "Interpolation Sides" begin
    function f(x, p, t)
        x.ctrl_u .- x
    end
    function f(dx, x, p, t)
        dx[1] = x.ctrl_u - x[1]
    end

    ctrl_f(e, x) = 0.85(e + 1.2x)

    x0 = CtrlSimTypeScalar([0.0], 0.0, 0.0)
    prob = ODEProblem{false}(f, x0, (0.0, 8.0))

    function ctrl_fun(int)
        e = 1 - int.u[1] # control error

        # pre-calculate values to avoid overhead in iteration over cache
        x_new = int.u.ctrl_x + e
        u_new = ctrl_f(e, x_new)

        # iterate over cache...
        if DiffEqBase.isinplace(int.sol.prob)
            for c in full_cache(int)
                c.ctrl_x = x_new
                c.ctrl_u = u_new
            end
        end
    end

    integrator = init(prob, Tsit5())

    # take 8 steps with a sampling time of 1s
    Ts = 1.0
    for i in 1:8
        ctrl_fun(integrator)
        DiffEqBase.step!(integrator, Ts, true)
    end

    sol = integrator.sol
    @test [u.ctrl_u for u in sol.u[2:end]] == [sol(t).ctrl_u for t in sol.t[2:end]]

    prob = ODEProblem{true}(f, x0, (0.0, 8.0))

    integrator = init(prob, Rosenbrock23())

    # take 8 steps with a sampling time of 1s
    Ts = 1.0
    for i in 1:8
        ctrl_fun(integrator)
        DiffEqBase.step!(integrator, Ts, true)
    end

    sol = integrator.sol
    @test [u.ctrl_u for u in sol.u[2:end]] == [sol(t).ctrl_u for t in sol.t[2:end]]
end
