import vtk
import numpy as np
from pathlib import Path
import concurrent.futures
import fire
import json

np.random.seed(0)

def otf1(dMin,dMax,dRange):
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AllowDuplicateScalarsOn()
    opacityTransferFunction.AddPoint(dMin, 0)
    opacityTransferFunction.AddPoint(0.02, 0)
    opacityTransferFunction.AddPoint(0.03, 0.02)
    opacityTransferFunction.AddPoint(dMax, 1)

    return opacityTransferFunction

def otf2(dMin,dMax,dRange):
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AllowDuplicateScalarsOn()
    opacityTransferFunction.AddPoint(dMin, 0)
    opacityTransferFunction.AddPoint(0.02, 0)
    opacityTransferFunction.AddPoint(0.03, 0.02)
    opacityTransferFunction.AddPoint(dMax, 1)

    return opacityTransferFunction

def gtf1(dMin,dMax,dRange):
    gradientTransferFunction = vtk.vtkPiecewiseFunction()
    gradientTransferFunction.AllowDuplicateScalarsOn()
    gradientTransferFunction.AddPoint(0, 0.002)
    gradientTransferFunction.AddPoint(dRange, 1)
    return gradientTransferFunction

def gtf2(dMin,dMax,dRange):
    gradientTransferFunction = vtk.vtkPiecewiseFunction()
    gradientTransferFunction.AllowDuplicateScalarsOn()
    gradientTransferFunction.AddPoint(0, 0.02)
    gradientTransferFunction.AddPoint(dRange, 1)
    return gradientTransferFunction

def render_image(path, daryName, dMin,dMax, theta, phi,color_points,out_file_path):

    log_d_min = np.log10(dMin)
    log_d_max = np.log10(dMax)
    log_range = log_d_max - log_d_min
    dRange = dMax - dMin

    # Create the standard renderer, render window and interactor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()

    # specify the data array in the file to process
    reader.GetOutput().GetPointData().SetActiveAttribute(daryName,0)

    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = otf1(dMin,dMax,dRange)

    # Create transfer mapping scalar value to color
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.SetScaleToLog10()
    colorTransferFunction.SetColorSpaceToLab()

    p_ind =0
    color_points_len = len(color_points)
    while p_ind < color_points_len:
        v = color_points[p_ind + 0]
        r = color_points[p_ind + 1]
        g = color_points[p_ind + 2]
        b = color_points[p_ind + 3]

        log_x = log_d_min + log_range * v
        x = 10** log_x

        colorTransferFunction.AddRGBPoint(x, r,g,b)
        p_ind += 4

    # gradient
    gradientTransferFunction = gtf1(dMin,dMax,dRange)
    
    # Create a camera
    camera = vtk.vtkCamera()
    camera.SetPosition(0.0, 900000.0, 11389735.996549446)
    camera.SetFocalPoint(0.0, 900000.0, 0.0)
    camera.SetViewUp(0, 1, 0)
    camera.Azimuth(theta)
    camera.Elevation(phi)

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetGradientOpacity(gradientTransferFunction)
    volumeProperty.ShadeOn()  ### on/off shader 
    volumeProperty.SetInterpolationTypeToLinear()

    # The mapper / ray cast function know how to render the data
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren.AddVolume(volume)
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    ren.SetBackground(0.5, 0.5, 0.5) # gray
    renWin.SetSize(256, 256)
    renWin.Render()

    # Write PNG Files
    window2ImageFilter = vtk.vtkWindowToImageFilter()
    window2ImageFilter.SetInput(renWin)
    window2ImageFilter.Update()

    
    if not out_file_path.parent.exists():
        out_file_path.parent.mkdir(parents=True,exist_ok=True)
    
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(out_file_path)
    writer.SetInputData(window2ImageFilter.GetOutput())
    writer.Write()
    print(out_file_path)

def rand_rotate(arr):
    rand_theta = 2 * np.pi * np.random.random()
    rand_phi = 2 * np.pi * np.random.random()

    rx = np.array([[1,0,0],
                   [0,np.cos(rand_phi),-np.sin(rand_phi)],
                   [0,np.sin(rand_phi),np.cos(rand_phi)]])
    ry = np.array([[np.cos(rand_theta),0,np.sin(rand_theta)],
                  [0,1,0],
                  [-np.sin(rand_theta),0,np.cos(rand_theta)]])
    return arr.dot(ry).dot(rx)

def spherical_to_cartesian(theta,phi):
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x,y,z

def cartesian_to_spherical(arr):
    x = arr[:,0]
    y = arr[:,1]
    z = arr[:,2]

    r = np.sqrt(x*x + y*y + z*z)
    phi = np.arccos( z/r)
    theta = np.arctan2(y, x)

    return theta,phi

def fib_sample(num):
    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(0, num)

    theta = 2 * np.pi * i / golden_ratio

    epsilon = 0.36
    phi = np.arccos(1 - 2*(i+epsilon)/(num-1+2*epsilon))

    arr = spherical_to_cartesian(theta,phi)
    arr = np.array(arr).T
    rotate_arr = rand_rotate(arr)
    
    r_theta, r_phi = cartesian_to_spherical(rotate_arr)
    
    return r_theta,r_phi

def sample_render(select_list,min_max_list,attr,sample_num,color_map,out_dir):
    
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True,exist_ok=True)
    
    # get data file list
    with open(select_list, 'r') as f:
        paths = f.read().splitlines()
    
    # Get min, max
    with open(min_max_list, 'r') as f:
        min_max_dict = json.load(f)

    with open(color_map,'r') as f:
        color_points = json.load(f)[0]['RGBPoints']


    # use multi process to accumlate
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for p in paths:
            p = Path(p.rstrip())

            theta, phi = fib_sample(sample_num)
            theta = np.rad2deg(theta)
            theta = (theta + 360) % 360
            phi = np.rad2deg(phi)
            phi -= 90

            for angle in zip(theta,phi):
                data_path = p
                dpath_parts = data_path.parts
                params = dpath_parts[-3]
                time = dpath_parts[-1].rstrip('.vti').split('_')[-1]

                d_min = min_max_dict[params]['min']
                d_max = min_max_dict[params]['max']

                out_file_path = out_dir / params / f'{time}_{angle[0]:.10f}_{angle[1]:.10f}.png'
                
                if not out_file_path.exists():
                    executor.submit(render_image,data_path, attr, d_min,d_max, angle[0],angle[1],color_points,out_file_path)

    with open(Path(out_dir,'total.txt'),'w') as f:
        for m in sorted(Path(out_dir).iterdir()):
            if m.is_dir():
                for t in sorted(m.iterdir()):
                    f.write(f'{t}\n')

if __name__ == "__main__":
    fire.Fire()