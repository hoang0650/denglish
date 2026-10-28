#!/bin/bash
set -euo pipefail
if [[ ${#AIM_RDP_PASSWORD} -lt 24 ]]; then echo 'A strong per-session RDP password is required' >&2; exit 1; fi
printf 'player:%s\n' "$AIM_RDP_PASSWORD" | chpasswd
unset AIM_RDP_PASSWORD
mkdir -p /home/player/game /home/player/Desktop /run/xrdp /run/dbus
printf '#!/bin/sh\nunset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR\nexec dbus-run-session startxfce4\n' > /home/player/.xsession
chmod 755 /home/player/.xsession
printf '[Desktop Entry]\nType=Application\nName=Game server files\nExec=thunar /home/player/game\nIcon=folder\n' > /home/player/Desktop/Game-Files.desktop
chown -R player:player /home/player
openssl req -x509 -newkey rsa:2048 -nodes -keyout /etc/xrdp/key.pem -out /etc/xrdp/cert.pem -days 7 -subj '/CN=AI Markets Linux Desktop' >/dev/null 2>&1
chown root:xrdp /etc/xrdp/key.pem && chmod 640 /etc/xrdp/key.pem
sed -i 's/^security_layer=.*/security_layer=tls/;s/^crypt_level=.*/crypt_level=high/;s/^certificate=.*/certificate=\/etc\/xrdp\/cert.pem/;s/^key_file=.*/key_file=\/etc\/xrdp\/key.pem/' /etc/xrdp/xrdp.ini
dbus-uuidgen --ensure
dbus-daemon --system --fork
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf
