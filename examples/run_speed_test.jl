@everywhere require("examples/speed_test.jl")
speed_test()

for i=1:5
    addprocs(2)
    @everywhere require("examples/speed_test.jl")
    println("Speed test using $(nprocs()) processors")
    speed_test()
end

 
