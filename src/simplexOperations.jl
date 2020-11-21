"""
    NCNBD.divide_simplex_by_longest_edge!(simplex_index::Int64, triangulation::NCNBD.Triangulation)

Divides a given simplex by longest edge (by essentially deleting it from the triangulation
and constructing two new simplices)

"""

function divide_simplex_by_longest_edge!(simplex_index::Int64, triangulation::NCNBD.Triangulation)

    simplex = triangulation.simplices[simplex_index]
    dimension = size(simplex.vertices, 1) - 1

    if dimension == 1
        new_simplices = divide_simplex_by_longest_edge_1D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    elseif dimension == 2
        new_simplices = divide_simplex_by_longest_edge_2D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    end

    return new_simplices
end

function divide_simplex_by_longest_edge_1D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    # get simplex and nlfunction
    simplex = triangulation.simplices[simplex_index]
    nlfunction = triangulation.ext[:nonlinearFunction]

    # Pre-allocate space for new simplices
    new_simplices = Vector{NCNBD.Simplex}(undef, 2)

    # determine midpoint of simplex (interval)
    midpoint = 0.5 * (simplex.vertices[1,1] + simplex.vertices[2,1])
    func_value = nlfunction.nonlinfunc_eval(midpoint)

    # create two new simplices and add them to the triangulation
    new_simplices[1] = NCNBD.Simplex(Array{Float64,2}(undef, 2, 1), Vector{Float64}(undef, 2), Inf, Inf)
    new_simplices[1].vertices[:,1] = [simplex.vertices[1,1], midpoint]
    new_simplices[1].vertice_values = [simplex.vertice_values[1], func_value]
    push!(triangulation.simplices, new_simplices[1] )

    new_simplices[2] = NCNBD.Simplex(Array{Float64,2}(undef, 2, 1), Vector{Float64}(undef, 2), Inf, Inf)
    new_simplices[2].vertices[:,1] = [midpoint, simplex.vertices[2,1]]
    new_simplices[2].vertice_values = [func_value, simplex.vertice_values[2]]
    push!(triangulation.simplices, new_simplices[2])

    # delete old simplex
    deleteat!(triangulation.simplices, simplex_index)

    return new_simplices
end

function divide_simplex_by_longest_edge_2D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    # get simplex and nlfunction
    simplex = triangulation.simplices[simplex_index]
    nlfunction = triangulation.ext[:nonlinearFunction]

    # Pre-allocate space for new simplices
    new_simplices = Vector{NCNBD.Simplex}(undef, 2)

    # define edges
    edges = [[1,2], [1,3], [2,3]]
    longest_edge = 0
    longest_edge_length = 0

    # determine distances between vertices
    for edge in edges
        vertex_1 = simplex.vertices[edge[1], :]
        vertex_2 = simplex.vertices[edge[2], :]
        dist = Distances.euclidean(vertex_1, vertex_2)

        if dist > longest_edge_length
            longest_edge = edge
            longest_edge_length = dist
        end
    end

    # determine the midpoint of the longest edge
    vertex_indices = [1, 2, 3]
    midpoint = Vector{Float64}(undef, 2)
    midpoint[1] = 0.5 * (simplex.vertices[longest_edge[1], 1] + simplex.vertices[longest_edge[2], 1])
    midpoint[2] = 0.5 * (simplex.vertices[longest_edge[1], 2] + simplex.vertices[longest_edge[2], 2])
    func_value = nlfunction.nonlinfunc_eval(midpoint[1], midpoint[2])

    # determine vertex which is not part of the longest edge
    non_edge_vertex = findall(x->typeof(x)==Nothing, indexin(vertex_indices, longest_edge))[1]

    # pre-allocate vertex space
    vertices = Array{Float64,2}(undef, 3, 2)

    # create two new simplices and add them to the triangulation
    # first simplex
    new_simplices[1] = NCNBD.Simplex(Array{Float64,2}(undef, 3, 2), Vector{Float64}(undef, 3), Inf, Inf)
    new_simplices[1].vertices[1, :] = simplex.vertices[non_edge_vertex, :]
    new_simplices[1].vertices[2, :] = simplex.vertices[longest_edge[1], :]
    new_simplices[1].vertices[3, :] = [midpoint[1] midpoint[2]]
    new_simplices[1].vertice_values = [simplex.vertice_values[non_edge_vertex], simplex.vertice_values[longest_edge[1]], func_value]
    push!(triangulation.simplices, new_simplices[1])

    # second simplex
    new_simplices[1] = NCNBD.Simplex(Array{Float64,2}(undef, 3, 2), Vector{Float64}(undef, 3), Inf, Inf)
    new_simplices[1].vertices[1, :] = simplex.vertices[non_edge_vertex, :]
    new_simplices[1].vertices[2, :] = simplex.vertices[longest_edge[2], :]
    new_simplices[1].vertices[3, :] = [midpoint[1] midpoint[2]]
    new_simplices[1].vertice_values = [simplex.vertice_values[non_edge_vertex], simplex.vertice_values[longest_edge[2]], func_value]
    push!(triangulation.simplices, new_simplices[2])

    # delete old simplex
    deleteat!(triangulation.simplices, simplex_index)

    return new_simplices
end

function pointInTriangle(vertices::Array{Float64,2}, point::Vector{Float64})

    @assert size(vertices, 1) == 3
    @assert size(vertices, 2) == 2
    @assert size(point, 1) == 2

    denominator = ((vertices[2,2] - vertices[3,2])*(vertices[1,1] - vertices[3,1]) + (vertices[3,1] - vertices[2,1])*(vertices[1,2] - vertices[3,2]))
    weight_1 = ((vertices[2,2] - vertices[3,2])*(point[1] - vertices[3,1]) + (vertices[3,1] - vertices[2,1])*(point[2] - vertices[3,2])) / denominator
    weight_2 = ((vertices[3,2] - vertices[1,2])*(point[1] - vertices[3,1]) + (vertices[1,1] - vertices[3,1])*(point[2] - vertices[3,2])) / denominator
    weight_3 = 1 - weight_1 - weight_2

    return 0 <= weight_1 && weight_1 <= 1 && 0 <= weight_2 && weight_2 <= 1 && 0 <= weight_3 && weight_3 <= 1
end

function pointInInterval(vertices::Array{Float64,2}, point::Float64)

    @assert size(vertices, 1) == 2
    @assert size(vertices, 2) == 1

    return vertices[1] <= point <= vertices[2]
end
