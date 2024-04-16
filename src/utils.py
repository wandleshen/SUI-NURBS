import imgui
import glfw
import torch
import vtk
import numpy as np
from vtk.util import numpy_support
from imgui.integrations.glfw import GlfwRenderer
from geomdl import NURBS, knotvector, tessellate


def impl_glfw_init(window_name="SUI-NURBS", width=1280, height=720):
    """
    Initialize a GLFW window and a VTK renderer.

    Parameters:
    window_name (str): The name of the window. Default is "SUI-NURBS".
    width (int): The width of the window. Default is 1280.
    height (int): The height of the window. Default is 720.

    Returns:
    window (GLFWwindow*): The created GLFW window.
    render_window (vtkExternalOpenGLRenderWindow): The VTK render window.
    renderer (vtkRenderer): The VTK renderer.
    """
    # Initialize the GLFW library
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    # Make the window's context current
    glfw.make_context_current(window)

    # Create a VTK renderer
    renderer = vtk.vtkRenderer()

    # Create a VTK render window
    render_window = vtk.vtkExternalOpenGLRenderWindow()
    render_window.SetSize(width, height)
    render_window.AddRenderer(renderer)

    return window, render_window, renderer


def draw_aabb(pts, renderer, isRed=True, isGreen=False):
    """
    Draw Axis-Aligned Bounding Boxes (AABBs) given their diagonal points.

    Parameters:
    pts (list or torch.Tensor): Diagonal points of the AABBs. Each element should be a pair of points.
    renderer (vtkRenderer): The VTK renderer.
    isRed (bool): If True, the AABBs will be colored red. Default is True.
    isGreen (bool): If True, the AABBs will be colored green. Default is False.

    Returns:
    actors (list): A list of vtkActor objects representing the AABBs.
    """
    # Initialize the list of actors
    actors = []

    def add_actor(pts):
        # Iterate over all given AABB diagonal pairs
        for min_point, max_point in zip(pts[:, 0], pts[:, 1]):
            # Create AABB box
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

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(box.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLighting(False)

            # Set the color of the actor
            if isRed:
                actor.GetProperty().SetColor(1.0, 0.0, 0.0)
            elif isGreen:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            else:
                actor.GetProperty().SetColor(0.0, 0.0, 1.0)

            # Add the actor to the renderer
            renderer.AddActor(actor)
            actors.append(actor)

    # Check if pts is a list or a numpy array
    if isinstance(pts, list):
        for pt in pts:
            add_actor(pt)
    else:
        add_actor(pts)

    return actors


def draw_surf(surf, renderer, isGreen=True):
    """
    Draw a surface given a NURBS surface object.

    Parameters:
    surf (object): A NURBS surface object.
    renderer (vtkRenderer): The VTK renderer.
    isGreen (bool): If True, the surface will be colored green. Default is True.

    Returns:
    actors (list): A list of vtkActor objects representing the surface.
    """
    # Add evaluation points
    plane_points = np.array(surf.evalpts)
    points = vtk.vtkPoints()
    polydata = vtk.vtkPolyData()

    # Add evaluation points to VTK points structure.
    for point in plane_points:
        points.InsertNextPoint(point.tolist())

    polydata.SetPoints(points)
    polydata.Modified()

    reconstruction = vtk.vtkSurfaceReconstructionFilter()
    reconstruction.SetInputData(polydata)
    reconstruction.Update()

    contours = vtk.vtkContourFilter()
    contours.SetInputConnection(reconstruction.GetOutputPort())
    contours.GenerateValues(1, 0, 0)
    output_port = contours.GetOutputPort()

    if surf.trims:
        appendFilter = vtk.vtkAppendPolyData()

        for trim in surf.trims:
            trimPtsArray = trim.evalpts
            trimPtsArray = surf.evaluate_list(trimPtsArray)
            # 创建多边形的点集和细胞
            trimPts = vtk.vtkPoints()
            trimPolys = vtk.vtkCellArray()

            # 添加点到 trimPts
            for pt in trimPtsArray:
                trimPts.InsertNextPoint(pt)

            # 创建多边形细胞数据
            poly = vtk.vtkPolygon()
            poly.GetPointIds().SetNumberOfIds(len(trimPtsArray))
            for idx, pt in enumerate(trimPtsArray):
                poly.GetPointIds().SetId(idx, idx)
            
            trimPolys.InsertNextCell(poly)

            # 创建单个的vtkPolyData对象
            singlePolyData = vtk.vtkPolyData()
            singlePolyData.SetPoints(trimPts)
            singlePolyData.SetPolys(trimPolys)

            # 添加到 appendFilter
            appendFilter.AddInputData(singlePolyData)

        # 所有多边形的数据进行合并
        appendFilter.Update()
        # 使用 vtkImplicitPolyDataDistance 将多边形数据转换为隐式函数
        implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
        implicitPolyDataDistance.SetInput(appendFilter.GetOutput())

        # 使用clipper执行裁剪操作
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(contours.GetOutputPort())
        clipper.SetClipFunction(implicitPolyDataDistance)  # 使用vtkImplicitPolyDataDistance作为裁剪函数
        # clipper.InsideOutOn()  # 如果需要的话裁剪外部
        output_port = clipper.GetOutputPort()

    polydata_mapper = vtk.vtkPolyDataMapper()
    polydata_mapper.SetInputConnection(output_port)
    polydata_mapper.ScalarVisibilityOff()

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
    """
    Draw a curve given a set of points.

    Parameters:
    curve (np.array): A set of points in the curve.
    renderer (vtkRenderer): The VTK renderer.

    Returns:
    actors (list): A list of vtkActor objects representing the curve(s).
    """
    actors = []

    def add_curve(curve):
        curve = curve.cpu().numpy()

        # Create a vtkPoints object and store the points in it
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(curve))

        # Create a polydata to store everything in
        polydata = vtk.vtkPolyData()

        spline = vtk.vtkParametricSpline()
        spline.SetPoints(points)

        function_source = vtk.vtkParametricFunctionSource()
        function_source.SetParametricFunction(spline)
        function_source.SetUResolution(100 * polydata.GetNumberOfPoints())
        function_source.Update()

        # Setup actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(function_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add the actor to the renderer
        renderer.AddActor(actor)
        actors.append(actor)

    if isinstance(curve, list):
        for c in curve:
            add_curve(c)
    else:
        add_curve(curve)
    return actors


def pan_camera(camera, right_amount, up_amount):
    """
    Pan the camera in the specified directions.

    Parameters:
    camera (vtkCamera): The camera to pan.
    right_amount (float): The amount to pan to the right.
    up_amount (float): The amount to pan upwards.
    """
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
    """
    Handle changes in the visualization.

    Parameters:
    index (int): The index of the change.
    key (str): The key of the actor in the actors dictionary.
    draw_func (function): The function to draw the actors.
    renderer (vtkRenderer): The VTK renderer.
    actors_dict (dict): A dictionary of actors.
    changes (list): A list of changes.
    checkboxes (list): A list of checkbox states.
    *args: Additional arguments to pass to the draw function.
    **kwargs: Additional keyword arguments to pass to the draw function.
    """
    if changes[index]:
        if checkboxes[index]:
            actors_dict[key] = draw_func(*args, renderer=renderer, **kwargs)
        else:
            for actor in actors_dict[key]:
                renderer.RemoveActor(actor)


def render(
    surf1,
    surf2,
    extract1,
    extract2,
    stripped1,
    stripped2,
    cluster1,
    curve,
    window_name="SUI-NURBS",
    width=1280,
    height=720,
):
    """
    Render the given NURBS surfaces and curves in a GLFW window.

    Parameters:
    surf1, surf2 (object): NURBS surface objects.
    extract1, extract2 (torch.Tensor): Tensors of intersected AABBs for the surfaces.
    stripped1, stripped2 (torch.Tensor): Tensors of stripped AABBs for the surfaces.
    cluster1 (torch.Tensor): Tensor of clustered AABBs for the first surface.
    curve (torch.Tensor): Curve points.
    window_name (str): The name of the window. Default is "SUI-NURBS".
    width (int): The width of the window. Default is 1280.
    height (int): The height of the window. Default is 720.
    """
    # Initialize the GLFW window and the VTK renderer
    window, render_window, renderer = impl_glfw_init(window_name, width, height)
    # Initialize ImGui and the GLFW renderer
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Reset the camera and set the background color
    renderer.ResetCamera()
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Initialize the last mouse position and the window size
    lastX, lastY = width // 2, height // 2
    windowSize = [width, height]

    # Define the mouse move event handler
    def on_mouse_move(window, xpos, ypos):
        global lastX, lastY
        if imgui.get_io().want_capture_mouse:
            return
        camera = renderer.GetActiveCamera()
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            deltaX = xpos - lastX
            deltaY = ypos - lastY
            # Get the current camera
            camera.Azimuth(-deltaX)
            camera.Elevation(deltaY)
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            deltaX = xpos - lastX
            deltaY = ypos - lastY
            # Scale
            deltaX /= windowSize[0]
            deltaY /= windowSize[1]
            # Pan the camera
            pan_camera(camera, -deltaX, deltaY)

        renderer.ResetCameraClippingRange()
        camera.OrthogonalizeViewUp()
        lastX = xpos
        lastY = ypos

    # Define the mouse button event handler
    def on_mouse_button(window, button, action, mods):
        global lastX, lastY
        if imgui.get_io().want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                (xpos, ypos) = glfw.get_cursor_pos(window)
                lastX = xpos
                lastY = ypos

    # Define the mouse scroll event handler
    def on_mouse_scroll(window, xoffset, yoffset):
        if imgui.get_io().want_capture_mouse:
            return
        camera = renderer.GetActiveCamera()
        if yoffset > 0:
            camera.Zoom(1.1)
        else:
            camera.Zoom(0.9)
        renderer.ResetCameraClippingRange()

    # Set the mouse event callbacks
    glfw.set_cursor_pos_callback(window, on_mouse_move)
    glfw.set_scroll_callback(window, on_mouse_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)

    # Initialize the checkboxes, changes and actors dictionary
    checkboxes = [False, False, True, True, False, False, False, False]
    changes = [False, False, False, False, False, False, False, False]
    actors_dict = {}

    # Draw the surfaces
    actors_dict["surf1"] = draw_surf(surf1, renderer)
    actors_dict["surf2"] = draw_surf(surf2, renderer, isGreen=False)

    # Main loop
    while not glfw.window_should_close(window):
        # Handle ImGui inputs and start a new ImGui frame
        impl.process_inputs()
        imgui.new_frame()
        imgui.begin("Dashboard", True)

        # Create ImGui checkboxes for the surfaces, curves and AABBs
        changes[0], checkboxes[0] = imgui.checkbox("AABB1", checkboxes[0])
        changes[1], checkboxes[1] = imgui.checkbox("AABB2", checkboxes[1])
        changes[2], checkboxes[2] = imgui.checkbox("Surface1", checkboxes[2])
        changes[3], checkboxes[3] = imgui.checkbox("Surface2", checkboxes[3])
        changes[4], checkboxes[4] = imgui.checkbox("Stripped1", checkboxes[4])
        changes[5], checkboxes[5] = imgui.checkbox("Stripped2", checkboxes[5])
        changes[6], checkboxes[6] = imgui.checkbox("Cluster Result", checkboxes[6])
        changes[7], checkboxes[7] = imgui.checkbox("Curve", checkboxes[7])

        # End the ImGui frame and render the ImGui output
        imgui.end()
        imgui.render()
        render_window.Render()

        # Handle changes
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
            "strip1",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            stripped1,
            isRed=False,
        )
        handle_changes(
            5,
            "strip2",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            stripped2,
            isRed=False,
        )
        handle_changes(
            6,
            "cluster1",
            draw_aabb,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            cluster1,
            isRed=True,
            isGreen=False,
        )
        handle_changes(
            7,
            "curve",
            draw_curve,
            renderer,
            actors_dict,
            changes,
            checkboxes,
            curve,
        )

        # Render the ImGui output and swap the GLFW buffers
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Shutdown ImGui and terminate GLFW
    impl.shutdown()
    glfw.terminate()


def gen_surface(ctrlpts, p, q, res=50, trim_curves=None):
    """
    Generate a NURBS surface given control points and degrees.

    Parameters:
    ctrlpts (list): Control points for the NURBS surface.
    p, q (int): Degrees of the surface in U and V direction.
    res (int): Resolution for the surface evaluation. Default is 50.

    Returns:
    surf (object): A NURBS surface object.
    """
    # Create a NURBS surface
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = p
    surf.degree_v = q

    # Set control points
    surf.ctrlpts2d = ctrlpts

    # Generate knot vectors
    surf.knotvector_u = knotvector.generate(p, len(ctrlpts))
    surf.knotvector_v = knotvector.generate(q, len(ctrlpts[0]))

    # Set evaluation delta
    surf.delta = 1.0 / res

    # Set trim curves
    if trim_curves:
        # Set surface tessellation component
        surf.tessellator = tessellate.TrimTessellate()

        # Set trim curves
        surf.trims = trim_curves

    # Evaluate surface points
    surf.evaluate()
    return surf


def extract_aabb(aabb, col):
    """
    Extract Axis-Aligned Bounding Boxes (AABBs) from a given tensor.

    Parameters:
    aabb (torch.Tensor): A tensor of AABBs.
    col (torch.Tensor): A tensor of indices to extract.

    Returns:
    extract (numpy.ndarray): Extracted AABBs.
    aabb (numpy.ndarray): Remaining AABBs after extraction.
    """
    # Extract the specified AABBs
    extract = aabb[col[:, 0], col[:, 1]]

    # Create a mask to remove the extracted AABBs
    mask = torch.ones(aabb.shape[0], aabb.shape[1], dtype=torch.bool)
    mask[col[:, 0], col[:, 1]] = False

    # Apply the mask to the AABB tensor
    aabb = aabb[mask].view(-1, 2, 3)

    # Convert the tensors to numpy arrays
    return extract.cpu().numpy(), aabb.cpu().numpy()
