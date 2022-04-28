import vtk
import numpy as np
from pathlib import Path
import concurrent.futures
import fire

np.random.seed(0)

def RenderImage(path, daryName, dMin,dMax, theta, phi,out_file_path):

    # Create the standard renderer, render window and interactor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data
    reader = vtk.vtkImageReader()
    reader.SetFileDimensionality(3)
    reader.SetDataScalarTypeToFloat()
    reader.SetDataByteOrderToLittleEndian()
    reader.SetDataExtent(0,63,0,63,0,63)
    reader.SetFileName(path)
    reader.Update()

    # specify the data array in the file to process
    reader.GetOutput().GetPointData().SetActiveAttribute(daryName,0)

    # Create transfer mapping scalar value to opacity
    dRange = dMax - dMin
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(dMin, 0.0)
    opacityTransferFunction.AddPoint(dMax, 1)

    # Create transfer mapping scalar value to color
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(dMin, 0.231373, 0.298039, 0.752941)
    colorTransferFunction.AddRGBPoint(dMin + (dRange/2), 0.865003, 0.865003, 0.865003)
    colorTransferFunction.AddRGBPoint(dMax , 0.705882, 0.0156863, 0.14902)

    # gradient
    gradientTransferFunction = vtk.vtkPiecewiseFunction()
    gradientTransferFunction.AddPoint(dMin, 0.5)
    gradientTransferFunction.AddPoint(dMax, 1)
    
    # Create a camera
    camera = vtk.vtkCamera()
    camera.SetPosition(31.5, 31.5, 252.80223286601725)
    camera.SetFocalPoint(31.5, 31.5, 31.5)
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

def sample_render(select_list,min_max_list,attr,sample_num,out_dir):
    
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True,exist_ok=True)
    
    # get data file list
    with open(select_list, 'r') as f:
        paths = f.read().splitlines()
    
    # Get min, max
    with open(min_max_list, 'r') as f:
        lines = f.read().split('\n')
        min = float( lines[0].split(':')[1] )
        max = float( lines[1].split(':')[1] )
    print(min, max)


    # use multi process to accumlate
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
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
                params = dpath_parts[-3].split('_')
                params = f'{params[1]}_{params[2]}_{params[3]}'
                time = dpath_parts[-1].rstrip('.bin')[-3:]

                out_file_path = out_dir / params / f'{time}_{angle[0]:.10f}_{angle[1]:.10f}.png'
                
                if not out_file_path.exists():
                    executor.submit(RenderImage,data_path, attr, min, max, angle[0],angle[1],out_file_path)
    
    with open(Path(out_dir,'total.txt'),'w') as f:
        for m in sorted(Path(out_dir).iterdir()):
            if m.is_dir():
                for t in sorted(m.iterdir()):
                    f.write(f'{t}\n')

if __name__ == "__main__":
    fire.Fire()