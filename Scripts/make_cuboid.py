import trimesh

# Create a cuboid with dimensions 5mm x 5mm x 3mm
cuboid = trimesh.creation.box(extents=[0.05, 0.05, 0.03])

# Export the cuboid to an STL file
cuboid.export('cuboid.stl')
