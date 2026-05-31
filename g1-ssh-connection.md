# Connecting to the Unitree G1 (Jetson Orin NX)

Quick reference for SSH'ing into the G1's onboard compute via direct ethernet.

## Network Layout

The G1 runs a fixed internal subnet: `192.168.123.0/24`. No DHCP — you have to set a static IP on your laptop.

| IP                  | Device                              |
| ------------------- | ----------------------------------- |
| `192.168.123.161`   | Development PC port (ethernet jack) |
| `192.168.123.164`   | Jetson Orin NX (main compute)       |
| `192.168.123.99`    | Motion control board                |
| `192.168.123.13/14` | Onboard MCUs                        |

The Jetson Orin NX at `.164` is what you want for SSH.

## Setup

### 1. Plug In

Connect ethernet to the port on the G1's waist/back. Not the head port.

### 2. Set Static IP on Laptop

Pick any unused address on the subnet. `.222` is safe.

**Linux (NetworkManager):**

```bash
# Find your wired interface
nmcli device status

# Create the connection profile
sudo nmcli connection add type ethernet con-name g1-direct ifname <eth_iface> \
    ipv4.method manual \
    ipv4.addresses 192.168.123.222/24

# Bring it up
sudo nmcli connection up g1-direct
```

**macOS:** System Settings → Network → Ethernet → Details → TCP/IP → Configure IPv4: Manually → `192.168.123.222`, subnet `255.255.255.0`.

**Windows:** Settings → Network → Ethernet → IP assignment: Manual → IPv4 → `192.168.123.222`, mask `255.255.255.0`.

### 3. SSH In

```bash
ssh unitree@192.168.123.164
```

Default password: `123` (change this if it hasn't been changed already).

## Quality-of-Life

### Passwordless login

```bash
ssh-copy-id unitree@192.168.123.164
```

### SSH config shortcut

Add to `~/.ssh/config`:

```
Host g1
    HostName 192.168.123.164
    User unitree
```

Then just: `ssh g1`

### Toggling the connection

```bash
# Disconnect from G1
sudo nmcli connection down g1-direct

# Reconnect
sudo nmcli connection up g1-direct
```

## Troubleshooting

| Symptom                                | Cause / Fix                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `ssh` hangs                            | No L2 link. Try `ping 192.168.123.164`. If that fails, check cable / port.                                   |
| `ping` works, `ssh` doesn't            | sshd config changed or service down on the Jetson.                                                           |
| `No route to host`                     | Static IP didn't apply. Re-run `nmcli connection up g1-direct` and verify with `ip addr show <eth_iface>`.   |
| Laptop got `192.168.123.161` from DHCP | Some other connection is conflicting — delete it or set this profile's `connection.autoconnect-priority` higher. |
| Permission denied                      | Wrong password or someone changed the user. Default is `unitree` / `123`.                                    |

## Useful Commands Once SSH'd In

```bash
# Check WiFi status (e.g. the wlx... USB dongle)
nmcli device status

# Check Jetson stats (temp, power, GPU/CPU usage)
sudo tegrastats

# Active services
systemctl --type=service --state=running
```
