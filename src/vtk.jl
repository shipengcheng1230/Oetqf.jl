const gmshcelltype2vtkcelltype = Dict(
    1 => VTKCellTypes.VTK_LINE,
    2 => VTKCellTypes.VTK_TRIANGLE,
    3 => VTKCellTypes.VTK_QUAD,
    4 => VTKCellTypes.VTK_TETRA,
    5 => VTKCellTypes.VTK_HEXAHEDRON,
)

"""
    gen_vtk_grid(t::Val{:BEMHex8Mesh}, mesh)

This function uses GMSH to read the Hex8 mesh and convert it into a VTK grid format
    for visualization in Paraview or similar tools.

## Arguments
- `t::Val{:BEMHex8Mesh}`: a value type indicating the mesh type
- `mesh`: the mesh object to be converted

## Returns
- A tuple containing the points and cells of the mesh in VTK format.
"""
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

"""
    gen_vtk_grid(mesh::RectOkadaMesh)  

This function creates a grid suitable for visualization in Paraview or similar tools.

## Arguments
- `mesh::RectOkadaMesh`: the rectangular Okada mesh object

## Returns
- A tuple containing the x, y, and z coordinates of the grid points.
"""
function gen_vtk_grid(mesh::RectOkadaMesh)
    mesh.dip ≈ 90.0 || error("An inclined plane requires unstructured mesh in Paraview!")
    xs = cat(map(x -> x[1], mesh.ax), mesh.ax[end][2], dims=1)
    ys = 0.0: 0.0
    zs = cat(map(x -> x[2], mesh.aξ), mesh.aξ[end][1], dims=1)
    (xs, ys, zs)
end

gen_vtk_grid(mesh::RectOkadaMesh, output) = vtk_grid(output, gen_vtk_grid(mesh)...)

"""
    gen_pvd(mf::RectOkadaMesh, mafile, solh5, tstr, ufstrs, uastrs, steps, output;
        tscale=365*86400, mafiletype=Val(:BEMHex8Mesh))
This function generates a Paraview collection file from the Okada mesh (fault plane) and the volume mesh, and solution data stored in an HDF5 file.

## Arguments
- `mf::RectOkadaMesh`: the rectangular Okada mesh object
- `mafile`: the mesh file for the BEM model
- `solh5`: the HDF5 file containing the solution data
- `tstr`: the name of the time data in the HDF5 file
- `ufstrs`: a vector of strings representing the names of the solution components (fault) in the HDF5 file
- `uastrs`: a vector of strings representing the names of the solution components (mantle) in the HDF5 file
- `steps`: a vector of integers representing the time steps to be included in the output
- `output`: the output directory for the Paraview collection file
- `tscale`: a scaling factor for the time data (default is 365*86400 seconds, which is one year)
- `mafiletype`: the type of the mesh file, default is `:BEMHex8Mesh`, which indicates a BEM mesh with Hex8 elements

## Returns
- A Paraview collection file containing the mesh and solution data at specified time steps.
"""
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