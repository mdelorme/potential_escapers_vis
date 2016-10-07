from vispy.visuals import mesh
import numpy as np

### Overriding IsoSurface shaders, we don't want that bad default lighting
mesh.shading_vertex_template = """
varying vec3 v_normal_vec;
varying vec3 v_light_vec;
varying vec3 v_eye_vec;

varying vec4 v_ambientk;
varying vec4 v_light_color;
varying vec4 v_base_color;


void main() {

    v_ambientk = $ambientk;
    v_light_color = $light_color;
    v_base_color = $base_color;


    vec4 pos_scene = $visual2scene($to_vec4($position));
    vec4 normal_scene = $visual2scene(vec4($normal, 1.0));
    vec4 origin_scene = $visual2scene(vec4(0.0, 0.0, 0.0, 1.0));

    normal_scene /= normal_scene.w;
    origin_scene /= origin_scene.w;

    vec3 normal = normalize(normal_scene.xyz - origin_scene.xyz);
    v_normal_vec = normal; //VARYING COPY

    vec4 pos_front = $scene2doc(pos_scene);
    pos_front.z += 0.01;
    pos_front = $doc2scene(pos_front);
    pos_front /= pos_front.w;

    vec4 pos_back = $scene2doc(pos_scene);
    pos_back.z -= 0.01;
    pos_back = $doc2scene(pos_back);
    pos_back /= pos_back.w;

    vec3 eye = normalize(pos_front.xyz - pos_back.xyz);
    v_eye_vec = eye; //VARYING COPY

    vec3 light = normalize($light_dir.xyz);
    v_light_vec = light; //VARYING COPY

    gl_Position = $transform($to_vec4($position));
}
"""

mesh.shading_fragment_template = """
varying vec3 v_normal_vec;
varying vec3 v_light_vec;
varying vec3 v_eye_vec;

varying vec4 v_ambientk;
varying vec4 v_light_color;
varying vec4 v_base_color;

void main() {


    //DIFFUSE
    float diffusek = dot(v_light_vec, v_normal_vec);
    //clamp, because 0 < theta < pi/2
    diffusek  = clamp(diffusek, 0.0, 1.0);
    vec4 diffuse_color = v_light_color * diffusek;
    //diffuse_color.a = 1.0;

    //SPECULAR
    //reflect light wrt normal for the reflected ray, then
    //find the angle made with the eye
    float speculark = dot(reflect(v_light_vec, v_normal_vec), v_eye_vec);
    speculark = clamp(speculark, 0.0, 1.0);
    //raise to the material's shininess, multiply with a
    //small factor for spread
    speculark = 20.0 * pow(speculark, 200.0) * 0.01;

    vec4 specular_color = v_light_color * speculark;


    gl_FragColor = 
       v_base_color * (v_ambientk + diffuse_color) + specular_color;

    //gl_FragColor = vec4(speculark, 0, 1, 1.0);


}
"""


#### Overriding mesh update_data to keep track of the light
light_dir = (1.0, 1.0, 1.0)
def _update_data(self):
    md = self.mesh_data
    # Update vertex/index buffers
    if self.shading == 'smooth' and not md.has_face_indexed_data():
        v = md.get_vertices()
        if v is None:
            return False
        if v.shape[-1] == 2:
            v = np.concatenate((v, np.zeros((v.shape[:-1] + (1,)))), -1)
        self._vertices.set_data(v, convert=True)
        self._normals.set_data(md.get_vertex_normals(), convert=True)
        self._faces.set_data(md.get_faces(), convert=True)
        self._index_buffer = self._faces
        if md.has_vertex_color():
            self._colors.set_data(md.get_vertex_colors(), convert=True)
        elif md.has_face_color():
            self._colors.set_data(md.get_face_colors(), convert=True)
        else:
            self._colors.set_data(np.zeros((0, 4), dtype=np.float32))
    else:
        v = md.get_vertices(indexed='faces')
        if v is None:
            return False
        if v.shape[-1] == 2:
            v = np.concatenate((v, np.zeros((v.shape[:-1] + (1,)))), -1)
        self._vertices.set_data(v, convert=True)
        if self.shading == 'smooth':
            normals = md.get_vertex_normals(indexed='faces')
            self._normals.set_data(normals, convert=True)
        elif self.shading == 'flat':
            normals = md.get_face_normals(indexed='faces')
            self._normals.set_data(normals, convert=True)
        else:
            self._normals.set_data(np.zeros((0, 3), dtype=np.float32))
        self._index_buffer = None
        if md.has_vertex_color():
            self._colors.set_data(md.get_vertex_colors(indexed='faces'),
                                  convert=True)
        elif md.has_face_color():
            self._colors.set_data(md.get_face_colors(indexed='faces'),
                                  convert=True)
        else:
            self._colors.set_data(np.zeros((0, 4), dtype=np.float32))
    self.shared_program.vert['position'] = self._vertices

    # Position input handling
    if v.shape[-1] == 2:
        self.shared_program.vert['to_vec4'] = mesh.vec2to4
    elif v.shape[-1] == 3:
        self.shared_program.vert['to_vec4'] = mesh.vec3to4
    else:
        raise TypeError("Vertex data must have shape (...,2) or (...,3).")

    # Color input handling
    # If non-lit shading is used, then just pass the colors
    # Otherwise, the shader uses a base_color to represent the underlying
    # color, which is then lit with the lighting model
    colors = self._colors if self._colors.size > 0 else self._color.rgba
    if self.shading is None:
        self.shared_program.vert[self._color_var] = colors

    # Shading
    if self.shading is None:
        self.shared_program.frag['color'] = self._color_var
    else:
        # Normal data comes via vertex shader
        if self._normals.size > 0:
            normals = self._normals
        else:
            normals = (1., 0., 0.)
            
        self.shared_program.vert['normal'] = normals
        self.shared_program.vert['base_color'] = colors

        # Additional phong properties
        self.shared_program.vert['light_dir'] = light_dir
        self.shared_program.vert['light_color'] = (0.1, 0.1, 0.1, 1.0)
        self.shared_program.vert['ambientk'] = (0.7, 0.7, 0.7, 1.0)
        
    self._data_changed = False

# Overriding
mesh.MeshVisual._update_data = _update_data
