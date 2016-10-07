import numpy as np
import sys, os
import OpenGL.GL as gl
from vispy import app, scene, visuals, gloo, io
from vispy.visuals import transforms
from vispy.visuals.line import line
from vispy.geometry.isosurface import isosurface

### Overriding mesh for the Isosurface, dimming the lighting ...
import mesh_override

### Globals
f = 1.0
rt = 10.0
rt2 = rt**2
rt3 = rt**3
npts = 129      # Points in every direction
scale = 0.157   # Scaling factor
data = np.loadtxt('orbit2.tab', dtype=np.float32) / scale
n_frames = data.shape[0]
cur_frame = 1
automated = False
render    = False
color_green = (0.0, 1.0, 0.0, 1.0)
color_red   = (1.0, 0.0, 0.0, 1.0)


### Jacobi surface equation (iso surface when jsurface(i, j, k) = 0)
def jsurfacef(x, y, z):
    return 2.0*rt3 + np.sqrt(x*x+y*y+z*z) * (x*x - z*z/f - 3.0*rt2)

def jsurface(i, j, k):
    x = (i - (npts - 1) * 0.5) * scale
    y = (j - (npts - 1) * 0.5) * scale
    z = (k - (npts - 1) * 0.5) * scale
    return jsurfacef(x, y, z)

def in_surface(x, y, z):
    m1 = (x < -rt*0.5)
    m2 = (x > rt*0.5)
    return (jsurfacef(x, y, z) <= 0 & m1 & m2)


#### Star shaders
vs_stars = '''
varying vec4 v_color;
void main (void)
{
    vec4 visual_pos = vec4($position, 1);
    vec4 doc_pos = $visual_to_doc(visual_pos);
    gl_Position = $doc_to_render(doc_pos);
    gl_PointSize = 10.0f;

    v_color = $color;
}'''

fs_stars = '''
varying vec4 v_color;
void main()
{
    float r = length(gl_PointCoord.xy - vec2(0.5,0.5));
    if (r > 0.5)
      discard;

    gl_FragColor = v_color;
}'''


class StarVisual(visuals.Visual):
    def __init__(self):
        # Init
        visuals.Visual.__init__(self, vcode=vs_stars, fcode=fs_stars)

        # Blending
        self.set_gl_state('opaque', cull_face=False)

        # Position
        self.vbo = gloo.VertexBuffer(np.zeros((1,3), dtype=np.float32))

        # Drawing params
        self._draw_mode = 'points'
        self.shared_program.vert['position'] = self.vbo

        self.freeze()
        

    def _prepare_transforms(self, view):
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert['visual_to_doc'] = tr.get_transform('visual', 'document')
        view_vert['doc_to_render'] = tr.get_transform('document', 'render')

    def _prepare_draw(self, view):
        pos = data[cur_frame, 1:4]
        self.vbo.set_data(pos.reshape(1,3))

        # Computing the Jacobi surface value
        if in_surface(pos[0]*scale, pos[1]*scale, pos[2]*scale):
            color = color_green
        else:
            color = color_red
            
        self.shared_program.vert['color'] = color
        
StarMesh = scene.visuals.create_visual_node(StarVisual)

class TrailVisual(line.LineVisual):
    def __init__(self, **kwargs):
        self.trail_length = 20000
        pos = np.zeros((1, 3), dtype=np.float32)
        line.LineVisual.__init__(self, connect='strip', method='gl', pos=pos, color=(1.0, 0.0, 0.0, 1.0), width = 2, **kwargs)
        
    def _prepare_draw(self, view):
        
        start_frame = max(cur_frame - self.trail_length, 0)
        pos = data[start_frame:cur_frame+1, 1:4]

        # Computing the Jacobi surface value
        mask = in_surface(pos[:,0]*scale, pos[:,1]*scale, pos[:,2]*scale)
        N = cur_frame - start_frame + 1
        color = np.repeat(np.array([color_red], dtype=np.float32), N, axis=0)

        Ns = np.sum(mask)
        color[mask] = np.repeat(np.array([color_green], dtype=np.float32), Ns, axis=0)
                
        self.set_data(pos=pos, color=color)

        line.LineVisual._prepare_draw(self, view)
        

TrailMesh = scene.visuals.create_visual_node(TrailVisual)

# The class holding all of the data
class JacobiDemoCanvas(scene.SceneCanvas):
    def __init__(self):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(1280, 720), bgcolor='w')
        self.unfreeze()
        
        self.view = self.central_widget.add_view()
        
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.visuals=[]

        ##### Jacobi iso-surface
        print('Generating Jacobi iso-surface')
        data = np.fromfunction(jsurface, (npts, npts, npts))
        self.surface = scene.visuals.Isosurface(data, level=1e-5,
                                                color=(0.5, 0.6, 1, 1.0), shading='smooth', parent=self.view.scene)
        self.surface.set_gl_state('opaque', depth_test=True, cull_face=True)
        self.surface.transform = scene.transforms.STTransform(translate=(-npts*0.5, -npts*0.5, -npts*0.5))
        
        # Forcing the update of the 

        print('Generating iso curves')
        cl = np.arange(0, npts, npts/2)
        colors = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), cl.shape[0], axis=0)
        vertices, tris = isosurface(self.surface._data, self.surface._level)
        
        self.x_wireframe = scene.visuals.Isoline(vertices=vertices, tris=tris, data=vertices[:, 0],
                                                 levels=cl, color_lev=colors, parent=self.view.scene)
        self.x_wireframe.set_gl_state('opaque', depth_test=True)
        self.x_wireframe.transform = scene.transforms.STTransform(translate=(-npts*0.5, -npts*0.5, -npts*0.5))

        self.y_wireframe = scene.visuals.Isoline(vertices=vertices, tris=tris, data=vertices[:, 1],
                                                 levels=cl, color_lev=colors, parent=self.view.scene)
        self.y_wireframe.set_gl_state('opaque', depth_test=True)
        self.y_wireframe.transform = scene.transforms.STTransform(translate=(-npts*0.5, -npts*0.5, -npts*0.5))

        self.z_wireframe = scene.visuals.Isoline(vertices=vertices, tris=tris, data=vertices[:, 2],
                                                 levels=cl, color_lev=colors, parent=self.view.scene)
        self.z_wireframe.set_gl_state('opaque', depth_test=True)
        self.z_wireframe.transform = scene.transforms.STTransform(translate=(-npts*0.5, -npts*0.5, -npts*0.5))
        
        ##### The star
        print('Creating star marker')
        self.star = StarMesh(parent=self.view.scene)
        self.star.set_gl_state(depth_test=False)

        print('Generating trail')
        self.trail = TrailMesh(parent=self.view.scene)
        self.trail.set_gl_state(depth_test=False)

        ##### Camera
        print('Camera setup')
        self.transitions = [5000, 8000, 13000, 16000]
        self.azimuths    = [30,  0,  0, 0]
        self.elevations  = [10, 90, 90, 0]
            
        self.cam = scene.TurntableCamera(elevation=self.elevations[0], azimuth=self.azimuths[0])
        self.cam.set_range((-20, 20), (-20, 20), (-20, 20))
        self.cam.depth_value = 10000.0
        self.cam._scale_factor = 90.0
        self.view.camera = self.cam

        
        print('General setup')
        ##### Misc
        self.anim_speed = 20

        self.visuals.append(self.surface)
        self.visuals.append(self.x_wireframe)
        self.visuals.append(self.y_wireframe)
        self.visuals.append(self.z_wireframe)
        self.visuals.append(self.star)
        self.visuals.append(self.trail)

        #if not automated:
        self.show()

        print('Init finished')

    def on_timer(self, event):
        self.surface.mesh_data_changed()
        
        global cur_frame
        cur_frame = cur_frame + self.anim_speed

        if render and cur_frame > n_frames:
            print('')
            print('All frames have been rendered to the folder render')
            print('To make a movie from these frames, use ffmpeg :')
            print('ffmpeg -framerate 24 -i render/img_%06d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p movie.mov')
            exit(0)
        else:
            cur_frame = max(cur_frame % n_frames, 1)
        

        # Camera movement
        if automated:
            cur_pos = 0
            # Which transition are we on
            while cur_pos < len(self.transitions) and cur_frame > self.transitions[cur_pos]:
                cur_pos += 1

                
            # If first or last, ignore
            if cur_pos not in (0, len(self.transitions)):
                x0 = self.transitions[cur_pos-1]
                x1 = self.transitions[cur_pos]
                a0 = self.azimuths[cur_pos-1]
                a1 = self.azimuths[cur_pos]
                e0 = self.elevations[cur_pos-1]
                e1 = self.elevations[cur_pos]

                x = float(cur_frame - x0) / float(x1 - x0)
                da = a1 - a0
                de = e1 - e0
                a = a0 + da * x
                e = e0 + de * x

                self.cam.elevation = e
                self.cam.azimuth   = a

        self.update()

        if render:
            sys.stdout.write('\rRendering frame {} out of {}'.format((cur_frame)/self.anim_speed+1, n_frames/self.anim_speed))
            sys.stdout.flush()
            self._draw_scene()
            img = gloo.util._screenshot()
            frame_id = str(cur_frame/self.anim_speed).rjust(6, '0')
            filename = 'render/img_{0}.png'.format(frame_id)
            io.imsave(filename, img[:, :, 0:3])

    def on_draw(self, event):
        self.context.set_clear_color(self.bgcolor)
        self.context.set_viewport(0, 0, *self.physical_size)
        self.context.clear()
        
        for vis in self.visuals:
            vis.draw()


if __name__ == '__main__':
    # Automated mode
    if len(sys.argv) > 1:
        if '--render' in sys.argv:
            render = True
            print('Starting rendering mode')

            # Caution with that
            if os.path.exists('render'):
                os.system('rm -rf render/*')

        if '--auto' in sys.argv:
            automated = True
            
    gl.glEnable( gl.GL_LINE_SMOOTH ) 
    gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST ) 

    canvas = JacobiDemoCanvas()

    print('Starting app')
    app.run()

    
