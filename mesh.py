import gmsh

geometry_file = "./freecad-sketch.brep"
dimension = 2

# Initialize gmsh
gmsh.initialize()
gmsh.model.add("rectangle_with_circle")

# load the BREP file
# brep files don't contain info about units, so scaling has to be applied
gmsh.option.setNumber("Geometry.OCCScaling", 1)

volumes = gmsh.model.occ.importShapes(geometry_file)

# run this code after every change in the mesh to see what changed
gmsh.model.occ.synchronize()

# Set mesh size (global refinement)
# The smaller the number, the finer the mesh. Adjust this value as needed.
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.8)

# Physical Surfaces, "Boundaries" in pyGIMLi,
# pgrp tag = 1 --> Free Surface | pgrp tag > 1 --> Mixed BC
gmsh.model.addPhysicalGroup(2, [1], 1, name="transmitter")
gmsh.model.addPhysicalGroup(2, [2], 2, name="blob")
gmsh.model.addPhysicalGroup(2, [3], 3, name="silicone")
gmsh.model.addPhysicalGroup(1, [1,9,8,7,6,10], 4, name="boundary")

# Generate the mesh and write the mesh file
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(dimension)

gmsh.fltk.run()

gmsh.write("rectangle_with_circle.msh")

# Finalize Gmsh
gmsh.finalize()
