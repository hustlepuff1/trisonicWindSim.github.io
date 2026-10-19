#!/usr/bin/env python3
"""Check the real 2D/3D page using an isolated Firefox Marionette session.

Start Firefox with --marionette and marionette.port=2829 in its user.js.
The optional baseline JSON is a captured pre-extension 2D run. Reports and
screenshots stay under the ignored build_nozzle_review directory by default.
"""
import argparse
import base64
import json
from pathlib import Path
import socket
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


class Browser:
    def __init__(self, port):
        self.socket = socket.create_connection(('127.0.0.1', port), 10)
        self.socket.settimeout(90)
        self.index = 0
        self.receive()

    def receive(self):
        size = bytearray()
        while True:
            ch = self.socket.recv(1)
            if not ch:
                raise ConnectionError('Browser connection closed')
            if ch == b':':
                break
            size.extend(ch)
        data = bytearray()
        while len(data) < int(size):
            data.extend(self.socket.recv(int(size) - len(data)))
        return json.loads(data)

    def call(self, name, params=None):
        self.index += 1
        message = json.dumps([0, self.index, name, params or {}]).encode()
        self.socket.sendall(str(len(message)).encode() + b':' + message)
        reply = self.receive()
        if reply[2]:
            raise RuntimeError(reply[2])
        result = reply[3]
        return result.get('value', result) if isinstance(result, dict) else result

    def js(self, script):
        return self.call('WebDriver:ExecuteScript', {'script': script, 'args': [], 'newSandbox': False, 'sandbox': None})

    def screenshot(self, path):
        data = self.call('WebDriver:TakeScreenshot', {'id': None, 'full': True, 'hash': False})
        path.write_bytes(base64.b64decode(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8020/build/index.html')
    parser.add_argument('--port', type=int, default=2829)
    parser.add_argument('--output', type=Path, default=ROOT / 'build_nozzle_review')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    b = Browser(args.port)
    report = {'url': args.url, 'browser': 'Firefox Marionette', 'checks': []}
    def check(name, ok, details=None):
        report['checks'].append({'name': name, 'passed': bool(ok), 'details': details})
        if not ok:
            raise AssertionError(name)
    try:
        b.call('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {'acceptInsecureCerts': True}}})
        b.call('WebDriver:SetWindowRect', {'width': 1500, 'height': 1100})
        b.call('WebDriver:Navigate', {'url': args.url + '?paused=1'})
        check('page ready without error', b.js('return document.body.dataset.ready==="true" && !document.getElementById("errorBox").textContent;'))
        for _ in range(240):
            state = b.js('return window.nozzleSimulatorDebug.getVolumeState();')
            if state['ready'] or state['error']:
                break
            time.sleep(.25)
        check('default 2D view automatically loads visible detailed profiles',
              state['ready'] and not state['error'] and state['view'] == '2d' and
              b.js('return document.getElementById("volumeProfiles").getBoundingClientRect().width>0;'), state)
        samples = b.js('''const d=window.nozzleSimulatorDebug,out=[];for(const m of [1.5,1.75,2,2.5,3,3.5]) {
          d.setCommand(m);d.step(2400);const f=d.getFields();out.push({state:d.getState(),longitudinal:Array.from(f.longitudinal),crossplane:Array.from(f.crossplane)});}return out;''')
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            check('2D fields and controller states exactly equal pre-extension baseline', samples == baseline,
                  {'operating_conditions': 6, 'field_values_compared': sum(len(r['longitudinal']) + len(r['crossplane']) for r in samples)})
        before = samples[-1]['state']
        b.js('window.nozzleSimulatorDebug.setView("3d"); return true;')
        state = b.js('return window.nozzleSimulatorDebug.getVolumeState();')
        check('actual nozzle-only assets loaded', state['ready'] and not state['error'], state)
        check('3D selection preserves controller state', before == b.js('return window.nozzleSimulatorDebug.getState();'))
        manifest = json.loads((ROOT / 'volume/manifest.json').read_text())
        check('nozzle-only training count and rendering dimensions',
              manifest['domain']['id'] == 'nozzle_x_le_0' and
              0 < state['training']['cell_count'] < 3533741 and
              state['training']['cell_count'] == manifest['training']['cell_count'] and
              state['training']['case_count'] == 27 and state['grid_shape'] == manifest['grid']['shape'])
        coordinates = np.fromfile(ROOT / 'volume/coordinates.f32', dtype='<f4').reshape(len(manifest['geometry_L_mm']), *manifest['grid']['shape'], 3)
        check('every exported geometry stops exactly at physical x=0',
              np.all(coordinates[..., 0] <= 0) and np.all(coordinates[:, -1, ..., 0] == 0))
        numerics = []
        mean = np.fromfile(ROOT / 'volume/mean.f32', dtype='<f4')
        modes = np.fromfile(ROOT / 'volume/modes.f32', dtype='<f4').reshape(manifest['rank'], -1)
        for target in [1.75, 2.5, 3.25]:
            s = b.js(f'const d=window.nozzleSimulatorDebug;d.setCommand({target});d.step(4000);return {{control:d.getState(),volume:d.getVolumeState()}};')
            v = s['volume']; L = s['control']['actual_L_mm']
            c = np.array([np.interp(L, manifest['path']['L_mm'], np.asarray(manifest['path']['coefficients'])[:, r]) for r in range(manifest['rank'])])
            field = mean.copy()
            for coefficient, mode in zip(c, modes):
                field[:] = field.astype(float) + coefficient * mode.astype(float)
            expected = {'min': float(field.min()), 'max': float(field.max()), 'sample_mean': float(np.mean(field.astype(float)))}
            error = max(abs(expected[k] - v['field_stats'][k]) for k in expected)
            check(f'3D numeric reconstruction and actual L at M{target}', error < 2e-6 and abs(v['actual_L_mm'] - L) < 1e-12, {'max_stat_error': error, 'update_ms': v['update_ms']})
            numerics.append(s)
        report['operating_states'] = numerics
        controls = b.js('''const d=window.nozzleSimulatorDebug;d.setVolumeSlice(0);const first=d.getVolumeState();
          d.setVolumeSlice(first.grid_shape[0]-1);const last=d.getVolumeState();
          const sym=document.getElementById('volumeSymmetry');sym.value='quarter';sym.dispatchEvent(new Event('change',{bubbles:true}));
          const quarter=d.getVolumeState();sym.value='full';sym.dispatchEvent(new Event('change',{bubbles:true}));
          document.getElementById('volumeTestSection').click();return {first,last,quarter,final:d.getVolumeState()};''')
        check('nozzle slices end exactly at x=0 and symmetry toggles',
              controls['first']['slice_x_m'] < -5.8 and abs(controls['last']['slice_x_m']) < 1e-8 and
              max(abs(v) for v in controls['last']['slice_x_range_m']) < 1e-8 and
              controls['final']['slice_index'] == controls['last']['slice_index'] and
              controls['final']['geometry_bounds'][0][1] <= 1e-8 and
              controls['quarter']['symmetry'] == 'quarter' and controls['final']['symmetry'] == 'full')
        b.screenshot(args.output / 'volume_3d.png')
        b.js('window.nozzleSimulatorDebug.setView("both");return true;')
        time.sleep(.15)
        check('combined view shows both fields', b.js('return document.getElementById("longCanvas").getBoundingClientRect().width>0 && document.getElementById("volumeCanvas").getBoundingClientRect().width>0;'))
        b.screenshot(args.output / 'volume_both.png')
        png = b.js('const c=window.nozzleSimulatorDebug.exportPNG();return {width:c.width,height:c.height,png:c.toDataURL("image/png").split(",")[1]};')
        check('3D PNG export', png['width'] == 1800 and png['height'] == 1080)
        (args.output / 'volume_export.png').write_bytes(base64.b64decode(png['png']))
        b.call('WebDriver:SetWindowRect', {'width': 550, 'height': 1000})
        time.sleep(.2)
        check('narrow layout has no horizontal overflow', b.js('return document.documentElement.scrollWidth<=innerWidth;'))
        b.screenshot(args.output / 'volume_mobile.png')
        check('no UI error after controls', b.js('return !document.getElementById("errorBox").textContent && !window.nozzleSimulatorDebug.getVolumeState().error;'))
        report['passed'] = True
    finally:
        (args.output / 'browser_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        b.call('WebDriver:DeleteSession')
        b.socket.close()
    print(json.dumps({'passed': True, 'checks': len(report['checks']), 'report': str(args.output / 'browser_validation.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
