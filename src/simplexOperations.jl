"""
    NCNBD.divide_simplex_by_longest_edge!(simplex_index::Int64, triangulation::NCNBD.Triangulation)

Divides a given simplex by longest edge (by essentially deleting it from the triangulation
and constructing two new simplices)

"""

function divide_simplex_by_longest_edge!(simplex_index::Int64, triangulation::NCNBD.Triangulation)

    dimension = size(simplex.vertices, 1) - 1

    if dimension == 1
        divide_simplex_by_longest_edge_1D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    elseif dimension == 2
        divide_simplex_by_longest_edge_2D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    end
end

function divide_simplex_by_longest_edge_1D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    # get simplex and nlfunction
    simplex = triangulation.simplices[simplex_index]
    nlfunction = triangulation.ext[:nlcfunction]

    # determine midpoint of simplex (interval)
    midpoint = 0.5 * (simplex.vertices[1,1] + simplex.vertices[2,1])
    func_value = nlfunction.nonlinfunc_eval(midpoint)

    # pre-allocate vertex space
    vertices = Array{Float64,2}(undef, 2, 1)

    # create two new simplices and add them to the triangulation
    vertices[1,1] = simplex.vertices[1,1]
    vertices[2,1] = midpoint
    vertice_values = [simplex.vertice_values[1], func_value]
    simplex_new = NCNBD.Simplex(vertices, vertice_values, Inf, Inf)
    push!(triangulation.simplices, simplex_new)

    vertices[1,1] = midpoint
    vertices[2,1] = simplex.vertices[2,1]
    vertice_values = [func_value, simplex.vertice_values[2]]
    simplex_new = NCNBD.Simplex(vertices, vertice_values, Inf, Inf)
    push!(triangulation.simplices, simplex_new)

    # delete old simplex
    deleteat!(triangulation.simplices, simplex_index)

end

function divide_simplex_by_longest_edge_2D!(simplex_index::Int64, triangulation::NCNBD.Triangulation)
    # get simplex and nlfunction
    simplex = triangulation.simplices[simplex_index]
    nlfunction = triangulation.ext[:nlcfunction]

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
    end

    # determine the midpoint of the longest edge
    midpoint = Vector{Float64}(undef, 2)
    midpoint[1] = 0.5 * (simplex.vertices[longest_edge[1], 1] + simplex.vertices[longest_edge[2], 1])
    midpoint[2] = 0.5 * (simplex.vertices[longest_edge[1], 2] + simplex.vertices[longest_edge[2], 2])
    func_value = nlfunction.nonlinfunc_eval(midpoint[1], midpoint[2])

    # determine vertex which is not part of the longest edge
    non_edge_vertex = findall(x->typeof(x)==Nothing, indexin(vertices, edge))

    # pre-allocate vertex space
    vertices = Array{Float64,2}(undef, 3, 2)

    # create two new simplices and add them to the triangulation
    # first simplex
    vertices[1, :] = [simplex.vertices[non_edge_vertex, :]]
    vertices[2, :] = [simplex.vertices[edge[1], :]]
    vertices[3, :] = [midpoint[1], midpoint[2]]
    vertice_values = [simplex.vertice_values[non_edge_vertex], simplex.vertice_values[edge[1]], func_value]
    simplex_new = NCNBD.Simplex(vertices, vertice_values, Inf, Inf)
    push!(triangulation.simplices, simplex_new)

    # second simplex
    vertices[1, :] = [simplex.vertices[non_edge_vertex, :]]
    vertices[2, :] = [simplex.vertices[edge[2], :]]
    vertices[3, :] = [midpoint[1], midpoint[2]]
    vertice_values = [simplex.vertice_values[non_edge_vertex], simplex.vertice_values[edge[2]], func_value]
    simplex_new = NCNBD.Simplex(vertices, vertice_values, Inf, Inf)
    push!(triangulation.simplices, simplex_new)

    # delete old simplex
    deleteat!(triangulation.simplices, simplex_index)
end
