"""Start the rented game under the same Linux account as the RDP desktop."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

UA = 'AIMarketsGameHosting/1.0 (https://aimarkets.vn)'
def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=120)

def metadata(url):
    with request(url) as r:
        return json.load(r)

def download(url, dest, checksum, algorithm):
    if not url.startswith('https://') or not checksum:
        raise ValueError('Verified HTTPS download required')
    h = hashlib.new(algorithm)
    with request(url) as response, open(dest, 'wb') as out:
        while block := response.read(1024 * 1024):
            h.update(block); out.write(block)
    if h.hexdigest() != checksum:
        Path(dest).unlink(missing_ok=True)
        raise ValueError('Game download checksum mismatch')

def minecraft(game):
    if os.environ.get('EULA', '').upper() != 'TRUE':
        raise ValueError('Minecraft EULA acceptance required')
    if game == 'minecraft-paper':
        data = metadata('https://fill.papermc.io/v3/projects/paper')
        versions = [v for group in data['versions'].values() for v in group if '-' not in v]
        asset = None
        for version in versions[:15]:
            builds = metadata(f'https://fill.papermc.io/v3/projects/paper/versions/{version}/builds')
            stable = next((b for b in builds if b['channel'] == 'STABLE'), None)
            if stable:
                asset = stable['downloads']['server:default']; break
        if not asset:
            raise ValueError('No stable Paper build available')
        download(asset['url'], 'server.jar', asset['checksums']['sha256'], 'sha256')
    else:
        manifest = metadata('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json')
        release = next(v for v in manifest['versions'] if v['id'] == manifest['latest']['release'])
        server = metadata(release['url'])['downloads']['server']
        download(server['url'], 'server.jar', server['sha1'], 'sha1')
    Path('eula.txt').write_text('eula=true\n')
    Path('server.properties').write_text('server-port=25565\nonline-mode=true\nmax-players=20\nview-distance=8\nenable-rcon=false\nmotd=AI Markets game server\n')
    memory = os.environ.get('MEMORY', '3G')
    if memory not in ('2G', '3G', '4G', '5G'):
        raise ValueError('Invalid game memory limit')
    os.execvp('java', ['java', '-Xms512M', '-Xmx' + memory, '-jar', 'server.jar', 'nogui'])

def main():
    Path('/home/player/game').mkdir(parents=True, exist_ok=True)
    os.chdir('/home/player/game')
    game = os.environ.get('AIM_GAME', 'desktop')
    if game in ('minecraft-java', 'minecraft-paper'):
        minecraft(game)
    elif game == 'terraria':
        shutil.copytree('/opt/tshock', 'tshock', dirs_exist_ok=True)
        os.chdir('tshock')
        os.execv('./TShock.Server', ['./TShock.Server', '-world', '/home/player/game/aimarkets.wld', '-autocreate', '1', '-worldname', 'AIMarkets', '-maxplayers', '8', '-port', '7777', '-configpath', '/home/player/game/config'])
    elif game == 'desktop':
        while True: time.sleep(60)
    else:
        raise ValueError('Unsupported game')

if __name__ == '__main__':
    main()
