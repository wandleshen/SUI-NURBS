import imgui
import glfw
import torch
import vtk
import numpy as np
from vtk.util import numpy_support
from imgui.integrations.glfw import GlfwRenderer
from geomdl import NURBS, knotvector


def impl_glfw_init(window_name="SUI-NURBS", width=1280, height=720):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    glfw.make_context_current(window)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkExternalOpenGLRenderWindow()
    render_window.SetSize(width, height)
    render_window.AddRenderer(renderer)

    return window, render_window, renderer


def draw_aabb(pts, renderer, isRed=True):
    # Iterate over all given AABB diagonal pairs, create and display each box.
    actors = []
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
        actor.GetProperty().SetLighting(False)
        if isRed:
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        else:
            actor.GetProperty().SetColor(0.0, 0.0, 1.0)

        # Add to renderer.
        renderer.AddActor(actor)
        actors.append(actor)
    return actors


def draw_surf(surf, renderer, isGreen=True):
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
    return [polydata_actor]


def draw_curve(curve, renderer):
    curve = curve.cpu().numpy()

    # Create a vtkPoints object and store the points in it
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(curve))

    # Create a cell array to store the lines in and add the lines to it
    lines = vtk.vtkCellArray()

    for i in range(curve.shape[0] - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()

    # Add the points to the dataset
    linesPolyData.SetPoints(points)

    # Add the lines to the dataset
    linesPolyData.SetLines(lines)

    # Setup actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(linesPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Add the actor to the renderer
    renderer.AddActor(actor)
    return [actor]


def pan_camera(camera, right_amount, up_amount):
    # Get the current position, focal point and view up vector of the camera
    position = camera.GetPosition()
    focal_point = camera.GetFocalPoint()
    view_up = camera.GetViewUp()

    # Convert tuples to numpy arrays
    position = np.array(position)
    focal_point = np.array(focal_point)
    view_up = np.array(view_up)

    # Compute the direction of projection or the view plane normal
    view_plane_normal = position - focal_point
    view_plane_normal = view_plane_normal / np.linalg.norm(view_plane_normal)

    # Compute the right vector (cross product of view up vector and view plane normal)
    right_vector = np.cross(view_up, view_plane_normal)
    right_vector = right_vector / np.linalg.norm(right_vector)

    # Compute the up vector (cross product of view plane normal and right vector, to ensure orthogonal)
    up_vector = np.cross(view_plane_normal, right_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)

    # Scale the pan movements to match the specified amounts
    right_movement = right_vector * right_amount
    up_movement = up_vector * up_amount

    # Calculate the new camera position and focal point
    new_position = position + right_movement + up_movement
    new_focal_point = focal_point + right_movement + up_movement

    # Update the camera's position and focal point
    camera.SetPosition(new_position.tolist())
    camera.SetFocalPoint(new_focal_point.tolist())
    camera.SetViewUp(
        view_up.tolist()
    )  # Ensure to reset the view up vector too, if it has been changed


def handle_changes(
    index, key, draw_func, renderer, actors_dict, changes, checkboxes, *args, **kwargs
):
    if changes[index]:
        if checkboxes[index]:
            actors_dict[key] = draw_func(*args, renderer=renderer, **kwargs)
        else:
            for actor in actors_dict[key]:
                renderer.RemoveActor(actor)


def render(
    aabb1,
    aabb2,
    surf1,
    surf2,
    extract1,
    extract2,
    cluster1,
    cluster2,
    curve,
    window_name="SUI-NURBS",
    width=1280,
    height=720,
):
    window, render_window, renderer = impl_glfw_init(window_name, width, height)
    imgui.create_context()
    impl = GlfwRenderer(window)

    renderer.ResetCamera()
    renderer.SetBackground(0.1, 0.2, 0.4)

    lastX, lastY = width // 2, height // 2
    windowSize = [width, height]

    # 鼠标移动事件处理函数
    def on_mouse_move(window, xpos, ypos):
        global lastX, lastY
        if imgui.get_io().want_capture_mouse:
            return
        camera = renderer.GetActiveCamera()
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            deltaX = xpos - lastX
            deltaY = ypos - lastY
            # 获取当前的相机
            camera.Azimuth(-deltaX)
            camera.Elevation(deltaY)
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            deltaX = xpos - lastX
            deltaY = ypos - lastY
            # 在当前渲染窗口尺寸取一定比例的移动
            deltaX /= windowSize[0]
            deltaY /= windowSize[1]
            # 水平和垂直平移视图
            pan_camera(camera, -deltaX, deltaY)

        renderer.ResetCameraClippingRange()
        camera.OrthogonalizeViewUp()
        lastX = xpos
        lastY = ypos

    # 鼠标按钮事件处理函数
    def on_mouse_button(window, button, action, mods):
        global lastX, lastY
        if imgui.get_io().want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                (xpos, ypos) = glfw.get_cursor_pos(window)
                lastX = xpos
                lastY = ypos

    # 设置滚轮事件的回调函数
    def on_mouse_scroll(window, xoffset, yoffset):
        if imgui.get_io().want_capture_mouse:
            return
        camera = renderer.GetActiveCamera()
        if yoffset > 0:
            camera.Zoom(1.1)
        else:
            camera.Zoom(0.9)
        renderer.ResetCameraClippingRange()

    glfw.set_cursor_pos_callback(window, on_mouse_move)
    glfw.set_scroll_callback(window, on_mouse_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)

    checkboxes = [False, False, True, True, False, False, False]
    changes = [False, False, False, False, False, False, False]
    actors_dict = {}

    actors_dict["surf1"] = draw_surf(surf1, renderer)
    actors_dict["surf2"] = draw_surf(surf2, renderer, isGreen=False)

    while not glfw.window_should_close(window):
        impl.process_inputs()
        imgui.new_frame()
        imgui.begin("Dashboard", True)

        changes[0], checkboxes[0] = imgui.checkbox("AABB1", checkboxes[0])
        changes[1], checkboxes[1] = imgui.checkbox("AABB2", checkboxes[1])
        changes[2], checkboxes[2] = imgui.checkbox("Surface1", checkboxes[2])
        changes[3], checkboxes[3] = imgui.checkbox("Surface2", checkboxes[3])
        changes[4], checkboxes[4] = imgui.checkbox("Cluster1", checkboxes[4])
        changes[5], checkboxes[5] = imgui.checkbox("Cluster2", checkboxes[5])
        changes[6], checkboxes[6] = imgui.checkbox("Curve", checkboxes[6])

        imgui.end()
        imgui.render()
        render_window.Render()

        handle_changes(
            0, "aabb1", draw_aabb, renderer, actors_dict, changes, checkboxes, aabb1
        )
        handle_changes(
            0,
            "ext1",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            extract1,
            isRed=False,
        )
        handle_changes(
            1, "aabb2", draw_aabb, renderer, actors_dict, changes, checkboxes, aabb2
        )
        handle_changes(
            1,
            "ext2",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            extract2,
            isRed=False,
        )
        handle_changes(
            2, "surf1", draw_surf, renderer, actors_dict, changes, checkboxes, surf1
        )
        handle_changes(
            3,
            "surf2",
            draw_surf,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            surf2,
            isGreen=False,
        )
        handle_changes(
            4,
            "cluster1",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            cluster1,
            isRed=False,
        )
        handle_changes(
            5,
            "cluster2",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            cluster2,
            isRed=False,
        )
        handle_changes(
            6,
            "curve",
            draw_curve,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            curve,
        )

        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        glfw.poll_events()

    impl.shutdown()
    glfw.terminate()


def gen_surface(ctrlpts, res=50):
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
    surf.delta = 1.0 / res

    # Evaluate surface points
    surf.evaluate()
    return surf


def extract_aabb(aabb, col):
    extract = aabb[col[:, 0], col[:, 1]]

    mask = torch.ones(aabb.shape[0], aabb.shape[1], dtype=torch.bool)
    mask[col[:, 0], col[:, 1]] = False
    aabb = aabb[mask].view(-1, 2, 3)

    return extract.cpu().numpy(), aabb.cpu().numpy()
