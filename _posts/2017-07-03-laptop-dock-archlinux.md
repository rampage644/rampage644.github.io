---
layout: post
title: Archlinux i3-wm on a docking station
date: 2017-07-03 19:30 +0300
comments: true
---

I'm using Archlinux with i3 tiling window manager. On the hardware side, this laptop sometimes sits on a docking station with external monitor attached to it. Periodically, I take it away from my desk to work in a different environment. Here are several tips'n'tricks I found useful.

# Monitor hotplugging

First thing that is really convenient is to automate monitor setup. It is either one laptop screen or a combination of laptop and external monitor.

There are lots of recipes on the internet that employ `udev` rules and custom shell scripts. I won't suggest any novel ideas andjust share my solution with description and some problems workarounds.

First, here is `udev` rule (place it into `/etc/udev/rules.d/*.conf):

```
KERNEL=="card0", ACTION=="change", SUBSYSTEM=="drm", ENV{DISPLAY}=":0", ENV{XAUTHORITY}="/home/USERNAME/.Xauthority", RUN+="/home/USERNAME/bin/hotplug
_monitor.sh"
```

Some things to note:

 1. Replace USERNAME with actual username
 1. Use `udevadm monitor` to monitor events and tweak `KERNEL` variable as needed (could be `card1`, for example)
 1. `XAuthority` file location could different, check your distro
 1. Call `udevadm control --reload-rules` after any changes to the file

Adding this rule will call specified script on every plug/unplug event. Here is my sample script:

```sh
#!/usr/bin/bash

export DISPLAY=:0
export XAUTHORITY=/home/USERNAME/.Xauthority

set -e
MONITOR='DP-3'

function wait_for_monitor {
    xrandr | grep $MONITOR | grep '\bconnected'
    while [ $? -ne 0 ]; do
            logger -t "waiting for 100ms"
            sleep 0.1
            xrandr | grep $MONITOR | grep '\bconnected'
    done
 }

EXTERNAL_MONITOR_STATUS=$(cat /sys/class/drm/card0-$MONITOR/status )
if [ $EXTERNAL_MONITOR_STATUS  == "connected" ]; then
    wait_for_monitor
    xrandr --output $MONITOR --auto --primary --output LVDS-1 --auto --left-of $MONITOR
    /home/USERNAME/bin/i3plug.py restore
else
    /home/USERNAME/bin/i3plug.py save
    xrandr --output $MONITOR --off
fi

feh --bg-scale /home/USERNAME/wallpaper.jpg
```

Pretty simple: it checks specified monitor status and applies either configuration depending on its connectivity. **Important** trick is `wait_for_monitor` function. I found that X server (or `xrandr`) sometimes significantly lags behind udev/kernel event. This function ensures `xrandr` actually sees externel monitor connected.

Another cool thing here is `i3plug.py`. Initially I found this script somewhere on the web and slightly modified it to my needs. It saves current i3 layout and can restore it some time later. Obviously, it saves the layout on monitor unplug (undocking) and restores when laptop again finds its place on the docking station.

```python
#!/usr/bin/env python
import os
import i3
import sys
import pickle


PATH = "/home/USERNAME/.i3/workspace_mapping"


def showHelp():
    print(sys.argv[0] + " [save|restore]")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        showHelp()
        sys.exit(1)

    if sys.argv[1] == 'save':
        pickle.dump(i3.get_workspaces(), open(PATH, "wb"))
    elif sys.argv[1] == 'restore':
        try:
            workspace_mapping = pickle.load(open(PATH, "rb"))
        except Exception:
            print("Can't find existing mappings...")
            sys.exit(1)

        for workspace in workspace_mapping:
            i3.msg('command', 'workspace %s' % workspace['name'])
            i3.msg('command', 'move workspace to output %s' % workspace['output'])
        for workspace in filter(lambda w: w['visible'], workspace_mapping):
            i3.msg('command', 'workspace %s' % workspace['name'])
    else:
        showHelp()
        sys.exit(1)
```

# Wired and wireless internet switch

Another useful addition is to use wired connection (with RJ45 cable attached to dock station) when using dock and WiFi without it. This is achieved with a help of `netctl` failover profile. [Archwiki article](https://wiki.archlinux.org/index.php/Netctl#Bonding) perfectly explains how to do that.

In short, install `ifenslave` package and enable `bonding`:

```sh
$ cat /etc/modules-load.d/bonding.conf
bonding

$ cat /etc/modprobe.d/bonding.conf
options bonding mode=active-backup miimon=100 primary=eth0 max_bonds=0
```

Create `failover` netctl profile (`cat /etc/netctl/failover`), tune wired and wireless interfaces names:

```
Description='A wired connection with failover to wireless'
Interface='bond0'
Connection=bond
BindsToInterfaces=('enp0s25' 'wlp3s0')
IP='dhcp'
```

Create `wpa_supplicant` service:

```sh
$ cat /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
ctrl_interface=/run/wpa_supplicant
update_config=1

network={
    ssid="SSID"
    psk=PSK
}
```


Finally, enable `netctl` and `wpa_supplicant@` template service. That's it.
