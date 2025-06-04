const gmshcelltype2vtkcelltype = Dict(
    1 => VTKCellTypes.VTK_LINE,
    2 => VTKCellTypes.VTK_TRIANGLE,
    3 => VTKCellTypes.VTK_QUAD,
    4 => VTKCellTypes.VTK_TETRA,
    5 => VTKCellTypes.VTK_HEXAHEDRON,
)

function gen_vtk_grid(::Val{:BEMHex8Mesh}, mesh)
    @gmsh_open mesh begin
        @addOption begin
            "General.Terminal", 0
        end
        nodes = gmsh.model.mesh.get_nodes()
        es = gmsh.model.mesh.get_elements(3, -1)
        nnode = gmsh.model.mesh.get_element_properties(es[1][1])[4]
        nume = length(es[2][1])
        conn = reshape(es[3][1], Int(nnode), :)
        celltype = gmshcelltype2vtkcelltype[es[1][1]]
        cells = [MeshCell(celltype, view(conn, :, i)) for i ∈ 1: nume]
        points = reshape(nodes[2], 3, :)
        return (points, cells)
    end
end

gen_vtk_grid(t::Val{:BEMHex8Mesh}, mesh, output) = vtk_grid(output, gen_vtk_grid(t, mesh)...)

function gen_vtk_grid(mesh::RectOkadaMesh)
    mesh.dip ≈ 90.0 || error("An inclined plane requires unstructured mesh in Paraview!")
    xs = cat(map(x -> x[1], mesh.ax), mesh.ax[end][2], dims=1)
    ys = 0.0: 0.0
    zs = cat(map(x -> x[2], mesh.aξ), mesh.aξ[end][1], dims=1)
    (xs, ys, zs)
end

gen_vtk_grid(mesh::RectOkadaMesh, output) = vtk_grid(output, gen_vtk_grid(mesh)...)

function gen_pvd(mf::RectOkadaMesh, mafile, solh5, tstr, ufstrs, uastrs, steps, output;
    tscale=365*86400, mafiletype=Val(:BEMHex8Mesh))

    HDF5.ishdf5(solh5) || error("Must provide a solution file of HDF5 format.")

    pvd = paraview_collection(output)
    v1 = gen_vtk_grid(mf)
    v2 = gen_vtk_grid(mafiletype, mafile)
    f = h5open(solh5, "r")
    td = open_dataset(f, tstr)
    ufs = map(x -> open_dataset(f, x), ufstrs)
    uas = map(x -> open_dataset(f, x), uastrs)
    padding = ndigits(length(steps))

    for i ∈ steps
        vtm = vtk_multiblock(output * lpad(i, padding, '0'))
        vtk1 = vtk_grid(vtm, v1...)
        vtk2 = vtk_grid(vtm, v2...)
        for (ufstr, uf) ∈ zip(ufstrs, ufs)
            vtk1[ufstr, VTKCellData()] = uf[(Colon() for _ ∈ 1: ndims(uf)-1)..., i]
        end
        for (uastr, ua) ∈ zip(uastrs, uas)
            # tranpose because for now the 1st dim is num_element, 2nd is components
            vtk2[uastr, VTKCellData()] = ua[(Colon() for _ ∈ 1: ndims(ua)-1)..., i]'
        end
        pvd[td[i] / tscale] = vtm
    end
    vtk_save(pvd)
end