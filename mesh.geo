// Define the rectangle dimensions
Lx = 1;  // Length in the x direction
Ly = 1;  // Length in the y direction
lc = 0.1; // Mesh size

// Define the corner points
Point(1) = {0, 0, 0, lc}; // Bottom-left corner
Point(2) = {Lx, 0, 0, lc}; // Bottom-right corner
Point(3) = {Lx, Ly, 0, lc}; // Top-right corner
Point(4) = {0, Ly, 0, lc}; // Top-left corner

// Define the edges of the rectangle
Line(1) = {1, 2}; // Bottom edge
Line(2) = {2, 3}; // Right edge
Line(3) = {3, 4}; // Top edge
Line(4) = {4, 1}; // Left edge

// Create a surface from the lines
Line Loop(5) = {1, 2, 3, 4}; // Loop for surface
Plane Surface(6) = {5};      // Create the surface

// Define physical groups for boundary conditions
Physical Curve("Bottom") = {1};  // Physical group for the bottom boundary
Physical Curve("Right") = {2};   // Physical group for the right boundary
Physical Curve("Top") = {3};     // Physical group for the top boundary
Physical Curve("Left") = {4};    // Physical group for the left boundary

// Physical group for the domain (the entire rectangle)
Physical Surface("Domain") = {6};

// Generate the triangular mesh
Mesh 2; // Generate a 2D mesh
