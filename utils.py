import vtk
import numpy as np
from geomdl import NURBS, knotvector
import torch


RES = 50


def render(aabb1, aabb2, surf1, surf2, col1, col2):
    # Create a renderer, render window, and interactor.
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    add_aabb(aabb1, renderer)
    add_aabb(aabb2, renderer)
    add_aabb(col1, renderer, isRed=False)
    add_aabb(col2, renderer, isRed=False)

    add_evalpts(surf1, renderer)
    add_evalpts(surf2, renderer, isGreen=False)

    # Set camera position, start rendering and interactor.
    renderer.ResetCamera()
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()


def add_aabb(pts, renderer, isRed=True):
    # Iterate over all given AABB diagonal pairs, create and display each box.
    for min_point, max_point in zip(pts[:, 0], pts[:, 1]):
        # Create AABB box.
        box = vtk.vtkCubeSource()
        box.SetBounds(
            min_point[0],
            max_point[0],
            min_point[1],
            max_point[1],
            min_point[2],
            max_point[2],
        )
        box.Update()

        # Create mapper and actor.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(box.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        if isRed:
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        else:
            actor.GetProperty().SetColor(0.0, 0.0, 1.0)

        # Add to renderer.
        renderer.AddActor(actor)


def add_evalpts(surf, renderer, isGreen=True):
    # Add evaluation points
    plane_points = np.array(surf.evalpts)
    points = vtk.vtkPoints()
    polydata = vtk.vtkPolyData()

    # Add evaluation points to VTK points structure.
    for point in plane_points:
        points.InsertNextPoint(point.tolist())

    # Create point-based Delaunay 2D grid object and mapper.
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(polydata)
    delaunay.Update()

    polydata.SetPoints(points)
    polydata.Modified()

    polydata_mapper = vtk.vtkPolyDataMapper()
    polydata_mapper.SetInputConnection(delaunay.GetOutputPort())

    polydata_actor = vtk.vtkActor()
    polydata_actor.SetMapper(polydata_mapper)
    if isGreen:
        polydata_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
    else:
        polydata_actor.GetProperty().SetColor(0.0, 1.0, 1.0)

    # Add to renderer.
    renderer.AddActor(polydata_actor)


def gen_surface(ctrlpts):
    # Create a NURBS surface
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    # Set control points
    surf.ctrlpts2d = ctrlpts

    surf.knotvector_u = knotvector.generate(3, len(ctrlpts))
    surf.knotvector_v = knotvector.generate(3, len(ctrlpts[0]))

    # Set evaluation delta
    surf.delta = 1.0 / RES

    # Evaluate surface points
    surf.evaluate()
    return surf


def extract_aabb(aabb, col):
    extract = aabb[col[:, 0], col[:, 1]]

    mask = torch.ones(aabb.shape[0], aabb.shape[1], dtype=torch.bool)
    mask[col[:, 0], col[:, 1]] = False
    aabb = aabb[mask].view(-1, 2, 3)

    return extract.cpu().numpy(), aabb.cpu().numpy()
