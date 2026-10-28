FROM ryshe/terraria:tshock-1.4.5.8-6.2.1 AS terraria
FROM mcr.microsoft.com/dotnet/runtime:9.0 AS dotnet
FROM eclipse-temurin:25-jre-noble
ENV DEBIAN_FRONTEND=noninteractive DOTNET_ROOT=/opt/dotnet DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
RUN apt-get update && apt-get install -y --no-install-recommends xrdp xorgxrdp xserver-xorg-core xfce4-session xfce4-panel xfwm4 xfdesktop4 xfce4-settings thunar xfce4-terminal dbus dbus-system-bus-common dbus-user-session dbus-x11 supervisor python3 openssl sudo nano procps libicu74 libssl3t64 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1001 --shell /bin/bash player && usermod -aG sudo player \
 && passwd -l root && mkdir -p /run/xrdp /run/dbus /var/log/supervisor \
 && printf 'allowed_users=anybody\nneeds_root_rights=no\n' > /etc/X11/Xwrapper.config
COPY --from=dotnet /usr/share/dotnet /opt/dotnet
COPY --from=terraria /tshock /opt/tshock
COPY entrypoint.sh /usr/local/bin/aim-desktop
COPY game.py /usr/local/bin/aim-game.py
COPY supervisord.conf /etc/supervisor/conf.d/aim-desktop.conf
RUN chmod 755 /usr/local/bin/aim-desktop && python3 -m py_compile /usr/local/bin/aim-game.py
EXPOSE 3389 25565 7777
ENTRYPOINT ["/usr/local/bin/aim-desktop"]
