"""Local web-based 3D Gaussian Splat viewer.

Usage:
    python scripts/viewer.py --ply output/scene/point_cloud/iteration_30000/point_cloud.ply
"""

import argparse
import http.server
import io
import json
import struct
import threading
import webbrowser
from pathlib import Path

import numpy as np
from plyfile import PlyData

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gaussian Splat Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #000; overflow: hidden; font-family: monospace; }
canvas { display: block; width: 100vw; height: 100vh; }
#info {
    position: absolute; top: 10px; left: 10px; color: #aaa; font-size: 12px;
    background: rgba(0,0,0,0.6); padding: 8px 12px; border-radius: 4px;
    pointer-events: none;
}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="info">Loading...</div>
<script>
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { antialias: false });
const info = document.getElementById('info');

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
resize();

const vsrc = `#version 300 es
precision highp float;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec2 u_focal;
uniform vec2 u_viewport;

in vec3 a_pos;
in vec3 a_col;
in float a_opacity;
in vec3 a_scale;
in vec4 a_quat;

out vec3 v_col;
out float v_opacity;
out vec2 v_conic;
out vec2 v_center;
out float v_radius;

mat3 quat_to_mat(vec4 q) {
    float x=q.x, y=q.y, z=q.z, w=q.w;
    return mat3(
        1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
        2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
        2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y)
    );
}

void main() {
    vec4 cam = u_view * vec4(a_pos, 1.0);
    float depth = -cam.z;
    vec4 clip = u_proj * cam;
    if (depth < 0.1) { gl_Position = vec4(0,0,-2,1); gl_PointSize = 0.0; return; }

    mat3 R = quat_to_mat(a_quat);
    vec3 s = a_scale;
    mat3 S = mat3(s.x,0,0, 0,s.y,0, 0,0,s.z);
    mat3 M = mat3(u_view) * R * S;
    mat3 cov3d = M * transpose(M);

    float fx = u_focal.x, fy = u_focal.y;
    mat2 J = mat2(fx/depth, 0, 0, fy/depth);
    mat2 cov2d = J * mat2(cov3d[0][0], cov3d[0][1], cov3d[1][0], cov3d[1][1]) * transpose(J);
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    float det = cov2d[0][0]*cov2d[1][1] - cov2d[0][1]*cov2d[1][0];
    if (det < 1e-6) { gl_Position = vec4(0,0,-2,1); gl_PointSize = 0.0; return; }

    float trace = cov2d[0][0] + cov2d[1][1];
    float eigenMax = 0.5*(trace + sqrt(max(trace*trace - 4.0*det, 0.0)));
    float radius = ceil(3.0 * sqrt(eigenMax));

    gl_Position = clip / clip.w;
    gl_PointSize = min(2.0 * radius, u_viewport.x);
    v_col = a_col;
    v_opacity = a_opacity;
    v_center = (gl_Position.xy * 0.5 + 0.5) * u_viewport;
    v_radius = radius;
    v_conic = vec2(1.0/max(cov2d[0][0],1e-6), 1.0/max(cov2d[1][1],1e-6));
}`;

const fsrc = `#version 300 es
precision highp float;
in vec3 v_col;
in float v_opacity;
in vec2 v_conic;
in vec2 v_center;
in float v_radius;
out vec4 fragColor;

void main() {
    vec2 d = gl_FragCoord.xy - v_center;
    float power = 0.5 * (v_conic.x * d.x * d.x + v_conic.y * d.y * d.y);
    if (power > 4.0) discard;
    float alpha = min(v_opacity * exp(-power), 0.99);
    if (alpha < 1.0/255.0) discard;
    fragColor = vec4(v_col * alpha, alpha);
}`;

function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(s));
    }
    return s;
}

const prog = gl.createProgram();
gl.attachShader(prog, createShader(gl.VERTEX_SHADER, vsrc));
gl.attachShader(prog, createShader(gl.FRAGMENT_SHADER, fsrc));
gl.linkProgram(prog);
gl.useProgram(prog);

const u_proj = gl.getUniformLocation(prog, 'u_proj');
const u_view = gl.getUniformLocation(prog, 'u_view');
const u_focal = gl.getUniformLocation(prog, 'u_focal');
const u_viewport = gl.getUniformLocation(prog, 'u_viewport');

gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
gl.enable(gl.PROGRAM_POINT_SIZE);
gl.disable(gl.DEPTH_TEST);

let numPoints = 0;
let rawData = null;
let sortedData = null;
let vbo = null;
let vao = null;
let sortWorker = null;
let sorting = false;
let lastSortCamPos = [Infinity, Infinity, Infinity];

const SORT_WORKER_SRC = `
self.onmessage = function(e) {
    const { positions, data, camX, camY, camZ, stride } = e.data;
    const n = positions.length / 3;
    const depths = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const dx = positions[i*3] - camX;
        const dy = positions[i*3+1] - camY;
        const dz = positions[i*3+2] - camZ;
        depths[i] = dx*dx + dy*dy + dz*dz;
    }
    const indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
    indices.sort((a, b) => depths[b] - depths[a]);
    const sorted = new Float32Array(n * stride);
    for (let i = 0; i < n; i++) {
        const src = indices[i] * stride;
        const dst = i * stride;
        for (let j = 0; j < stride; j++) sorted[dst+j] = data[src+j];
    }
    self.postMessage({ sorted }, [sorted.buffer]);
};`;

function createSortWorker() {
    const blob = new Blob([SORT_WORKER_SRC], { type: 'application/javascript' });
    return new Worker(URL.createObjectURL(blob));
}

function requestSort() {
    if (sorting || !rawData || numPoints === 0) return;
    const dx = camPos[0]-lastSortCamPos[0], dy = camPos[1]-lastSortCamPos[1], dz = camPos[2]-lastSortCamPos[2];
    if (dx*dx+dy*dy+dz*dz < 0.01 * camDist * camDist * 0.0001) return;
    sorting = true;
    const positions = new Float32Array(numPoints * 3);
    for (let i = 0; i < numPoints; i++) {
        positions[i*3] = rawData[i*14];
        positions[i*3+1] = rawData[i*14+1];
        positions[i*3+2] = rawData[i*14+2];
    }
    sortWorker.postMessage({
        positions, data: rawData, camX: camPos[0], camY: camPos[1], camZ: camPos[2], stride: 14
    }, [positions.buffer]);
}

async function loadData() {
    info.textContent = 'Fetching splat data...';
    const resp = await fetch('/data.bin');
    const buf = await resp.arrayBuffer();
    rawData = new Float32Array(buf);
    numPoints = rawData.length / 14;
    sortedData = new Float32Array(rawData);
    info.textContent = `${numPoints.toLocaleString()} Gaussians`;

    const stride = 14 * 4;

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, sortedData, gl.DYNAMIC_DRAW);

    const bind = (name, size, offset) => {
        const loc = gl.getAttribLocation(prog, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset * 4);
    };
    bind('a_pos', 3, 0);
    bind('a_col', 3, 3);
    bind('a_opacity', 1, 6);
    bind('a_scale', 3, 7);
    bind('a_quat', 4, 10);

    sortWorker = createSortWorker();
    sortWorker.onmessage = function(e) {
        sortedData = e.data.sorted;
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, sortedData, gl.DYNAMIC_DRAW);
        lastSortCamPos = [...camPos];
        sorting = false;
        info.textContent = `${numPoints.toLocaleString()} Gaussians (sorted)`;
    };
    requestSort();
}

// Camera (will be set from scene_info)
let camPos = [0, 0, 5];
let camTarget = [0, 0, 0];
let rotX = 0, rotY = 0;
let camDist = 5;

function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function normalize(v) { const l=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }
function sub(a,b) { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function dot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

function lookAt(eye, center, up) {
    const f = normalize(sub(center, eye));
    const s = normalize(cross(f, up));
    const u = cross(s, f);
    return new Float32Array([
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -dot(s,eye), -dot(u,eye), dot(f,eye), 1
    ]);
}

function perspective(fov, aspect, near, far) {
    const f = 1/Math.tan(fov/2);
    const nf = 1/(near-far);
    return new Float32Array([
        f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0
    ]);
}

let dragging = false, lastX, lastY, panMode = false;
canvas.addEventListener('mousedown', e => { dragging=true; lastX=e.clientX; lastY=e.clientY; panMode=e.button===2||e.shiftKey; });
canvas.addEventListener('mouseup', () => dragging=false);
canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX-lastX, dy = e.clientY-lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (panMode) {
        const speed = camDist * 0.002;
        const f = normalize(sub(camTarget, camPos));
        const s = normalize(cross(f, [0,1,0]));
        const u = cross(s, f);
        camTarget[0] -= (s[0]*dx + u[0]*dy)*speed;
        camTarget[1] -= (s[1]*dx + u[1]*dy)*speed;
        camTarget[2] -= (s[2]*dx + u[2]*dy)*speed;
    } else {
        rotY += dx * 0.005;
        rotX += dy * 0.005;
        rotX = Math.max(-Math.PI/2+0.01, Math.min(Math.PI/2-0.01, rotX));
    }
});
canvas.addEventListener('wheel', e => { camDist *= 1 + e.deltaY*0.001; camDist = Math.max(0.1, camDist); e.preventDefault(); }, {passive:false});
canvas.addEventListener('contextmenu', e => e.preventDefault());

function render() {
    camPos = [
        camTarget[0] + camDist * Math.cos(rotX) * Math.sin(rotY),
        camTarget[1] + camDist * Math.sin(rotX),
        camTarget[2] + camDist * Math.cos(rotX) * Math.cos(rotY)
    ];

    requestSort();

    const fov = Math.PI/3;
    const aspect = canvas.width / canvas.height;
    const fy = canvas.height / (2 * Math.tan(fov/2));
    const fx = fy;

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (numPoints > 0) {
        gl.uniformMatrix4fv(u_proj, false, perspective(fov, aspect, 0.1, 1000));
        gl.uniformMatrix4fv(u_view, false, lookAt(camPos, camTarget, [0,1,0]));
        gl.uniform2f(u_focal, fx, fy);
        gl.uniform2f(u_viewport, canvas.width, canvas.height);
        gl.drawArrays(gl.POINTS, 0, numPoints);
    }
    requestAnimationFrame(render);
}

async function init() {
    const siResp = await fetch('/scene_info');
    const si = await siResp.json();
    camTarget = si.center || [0,0,0];
    camDist = (si.extent || 10) * 0.8;
    rotY = Math.PI * 0.25;
    rotX = -0.3;
    await loadData();
    render();
}
init();
</script>
</body>
</html>"""


def load_ply_to_binary(ply_path: str) -> tuple[bytes, int, dict]:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    n = len(v["x"])

    pos = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)

    C0 = 0.28209479177387814
    if "f_dc_0" in v.data.dtype.names:
        col = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)
        col = col * C0 + 0.5
    elif "red" in v.data.dtype.names:
        col = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32) / 255.0
    else:
        col = np.ones((n, 3), dtype=np.float32) * 0.5
    col = np.clip(col, 0, 1)

    if "opacity" in v.data.dtype.names:
        opacity = 1.0 / (1.0 + np.exp(-np.array(v["opacity"], dtype=np.float32)))
    else:
        opacity = np.ones(n, dtype=np.float32)

    if "scale_0" in v.data.dtype.names:
        scales = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]).astype(np.float32)
        scales = np.exp(scales)
    else:
        scales = np.ones((n, 3), dtype=np.float32) * 0.01

    if "rot_0" in v.data.dtype.names:
        quats = np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]).astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats /= np.maximum(norms, 1e-8)
    else:
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0

    # Pack: pos(3) + col(3) + opacity(1) + scale(3) + quat(4) = 14 floats per point
    data = np.hstack([pos, col, opacity[:, None], scales, quats]).astype(np.float32)
    scene_info = {
        "center": pos.mean(axis=0).tolist(),
        "extent": float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))),
    }
    return data.tobytes(), n, scene_info


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    ply_data = b""
    num_points = 0
    scene_info = {}

    def do_GET(self):
        if self.path == "/data.bin":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", len(self.ply_data))
            self.end_headers()
            self.wfile.write(self.ply_data)
        elif self.path == "/scene_info":
            body = json.dumps(self.scene_info).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            html_bytes = HTML.encode()
            self.send_header("Content-Length", len(html_bytes))
            self.end_headers()
            self.wfile.write(html_bytes)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Gaussian splat viewer")
    parser.add_argument("--ply", required=True, help="Path to PLY file")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    print(f"Loading {args.ply}...")
    ViewerHandler.ply_data, n, ViewerHandler.scene_info = load_ply_to_binary(args.ply)
    ViewerHandler.num_points = n
    print(f"  {n:,} Gaussians ({len(ViewerHandler.ply_data) / 1024 / 1024:.1f} MB)")

    server = http.server.HTTPServer(("0.0.0.0", args.port), ViewerHandler)
    url = f"http://localhost:{args.port}"
    print(f"Viewer running at {url}")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    server.serve_forever()
