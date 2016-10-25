# Potential Escapers
Visualisation of a potential escaper in a cluster in tidal interaction with a galaxy.

# Requirements :
 - Vispy version >= 0.5
 - Python 2.7
 - Numpy
 - PyOpenGL
 
# License : MIT License

Copyright (C) 25/10/2016 Maxime Delorme
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Howto use :

The interactive mode is ran using :
```
python jacobi_surface.py
```
 
There is an automated camera transformation that was made for the production of the "movie" :
```
python jacobi_surface.py --auto
```

If you want to automate the camera you will find all the parameters in jacoby_surface.py :

```python
self.transitions = [5000, 8000, 13000, 16000]
self.azimuths    = [30,  0,  0, 0]
self.elevations  = [10, 90, 90, 0]
```

`transitions` indicates the key frame at which a certain value of elevation/azimuth fo the camera must be taken. In between keyframes, the system linearly interpolates.

Finally, you can render the frames to a folder by using the `--render` flag :

```
python jacoby_surface.py --auto --render
```
 
