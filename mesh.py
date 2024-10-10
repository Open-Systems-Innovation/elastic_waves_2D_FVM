import gmsh
import sys

gmsh.initialize()

gmsh.model.add("rectangular_mesh_with_tags")
lc = 0.02
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(3, 2, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)

gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 1, name="boundary")
gmsh.model.addPhysicalGroup(2, [1], 2, name="surface")

# We can then generate a 2D mesh...
gmsh.model.mesh.generate(2)

# ... and save it to disk
gmsh.write("mesh.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
