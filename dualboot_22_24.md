# Dual-Boot Guide: Ubuntu 22.04 LTS + Ubuntu 24.04 LTS

Step-by-step procedure to add Ubuntu 22.04.5 LTS alongside the existing Ubuntu 24.04.4 LTS on a single NVMe drive, using the prepared SanDisk USB stick.

---

## 0. System inventory (captured before starting)

| Item | Value |
|---|---|
| Current OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Disk | NVMe `WD_BLACK SN7100 1TB` — `/dev/nvme0n1` (931.5 GB) |
| Partitions before | `nvme0n1p1` = 1 GB EFI (vfat), `nvme0n1p2` = 930.5 GB ext4 root |
| Space used on root | ~103 GB of 915 GB |
| Firmware | UEFI |
| RAM | 62 GB |
| Installer USB | SanDisk Ultra Flair 28.7 GB — Ubuntu 22.04.5 LTS amd64 with persistence (`/dev/sda`) |
| Board | ASUS (boot menu: F8 at power-on) |

---

## 1. Target partition layout

Final layout after the procedure:

| # | Device | Size | FS | Mount | Purpose |
|---|---|---|---|---|---|
| 1 | `nvme0n1p1` | 1 GB | vfat | `/boot/efi` | EFI System Partition (shared, **do NOT reformat**) |
| 2 | `nvme0n1p2` | **550 GB** | ext4 | `/` (in 24.04) | Existing Ubuntu 24.04 root, shrunk |
| 3 | `nvme0n1p3` | **250 GB** | ext4 | `/` (in 22.04) | New Ubuntu 22.04 root |
| 4 | `nvme0n1p4` | ~130 GB | ext4 | `/data` (optional) | Shared data partition (optional but recommended) |

Total: 1 + 550 + 250 + 130 ≈ 931 GB.

GParted uses MiB. Type these values when prompted:

- Shrink `nvme0n1p2` new size: **563200 MiB** (= 550 GB)
- New 22.04 partition: **256000 MiB** (= 250 GB)
- Optional shared `/data`: rest of the unallocated space

---

## 2. Pre-flight checklist (do these BEFORE rebooting)

### 2.1 Back up data — mandatory

Resize operations are usually safe, but a power loss mid-resize destroys the filesystem. Back up to an external USB drive or cloud.

```bash
# Example: rsync home to an external drive mounted at /media/backup
rsync -aAXvh --info=progress2 \
    --exclude='.cache' --exclude='snap' --exclude='.local/share/Trash' \
    /home/ubuntu-alain-ethrc/ /media/backup/home-backup/
```

Also export browser bookmarks, SSH keys (`~/.ssh/`), GPG keys (`gpg --export-secret-keys`), and any database dumps you care about.

### 2.2 Note current EFI boot entries (so you can compare after)

```bash
efibootmgr -v > ~/Documents/efibootmgr_before.txt
lsblk -o NAME,SIZE,FSTYPE,UUID,MOUNTPOINT > ~/Documents/lsblk_before.txt
sudo blkid > ~/Documents/blkid_before.txt
```

### 2.3 Make sure you have a power supply you trust

A laptop on battery is risky; a desktop on a UPS is ideal. **Do not run a resize on an unstable power source.**

### 2.4 Confirm the USB stick is the 22.04 installer

Already verified: it shows label `Ubuntu 22.04.5 LTS amd64` on `/dev/sda1`. Safe to use.

---

## 3. Boot the USB live session

1. Reboot.
2. At the ASUS splash, press **F8** (boot menu) — or **F2/Del** for full BIOS if F8 doesn't work.
3. Select the SanDisk USB (UEFI entry, NOT the legacy one — pick the one prefixed with "UEFI:").
4. At the GRUB menu, choose **"Try or Install Ubuntu"**.
5. At the welcome screen, click **"Try Ubuntu"** — **NOT "Install Ubuntu" yet.**

You should now be on a live desktop running from the USB.

---

## 4. Shrink the existing Ubuntu 24.04 root

In the live session, open a terminal:

```bash
sudo apt update
sudo apt install -y gparted
sudo gparted
```

In GParted:

1. Top-right device dropdown → select **`/dev/nvme0n1`**.
2. Confirm `nvme0n1p2` is **not mounted** (no key icon). If it is, right-click → Unmount.
3. Right-click `nvme0n1p2` → **Resize/Move**.
4. In "New size (MiB)", enter **563200** (550 GB). Leave "Free space preceding" at 0.
5. Click **Resize/Move** (the dialog button).
6. Top toolbar → **Apply All Operations** (green check) → confirm.
7. Wait — this takes 10–30 minutes depending on data. **Do not power off or interrupt.**

When done, you should see ~381 GB of unallocated space after `nvme0n1p2`.

**Sanity check before continuing:**

```bash
sudo e2fsck -f /dev/nvme0n1p2
```

It should report the filesystem clean.

Close GParted.

---

## 5. Run the Ubuntu 22.04 installer

Double-click **"Install Ubuntu 22.04 LTS"** on the desktop.

Walk through the wizard:

- Language: your choice
- Keyboard: your choice
- **Updates and other software**: untick "Download updates while installing" (faster, do it after first boot)
- Installation type: **Something else** ← critical, do NOT pick "Install alongside" or "Erase disk"

At the partitioning screen you'll see the disk and partitions. Configure:

### 5.1 EFI partition — keep as-is

- Click `/dev/nvme0n1p1` → **Change…**
- "Use as": **EFI System Partition**
- **Format: UNCHECKED** ← critical. Formatting this nukes 24.04's bootloader.
- OK.

### 5.2 24.04 root — leave alone

- Do NOT touch `/dev/nvme0n1p2`. No changes. No format. No mount point.

### 5.3 New 22.04 root partition

- Click the **"free space"** entry → click **+**
- Size: **256000** MiB
- Type: **Primary** (or Logical — either works on GPT)
- Location: **Beginning of this space**
- Use as: **Ext4 journaling file system**
- Mount point: **`/`**
- OK.

### 5.4 (Optional) Shared data partition

- Click remaining "free space" → click **+**
- Size: all remaining
- Use as: **Ext4 journaling file system**
- Mount point: leave **empty** for now (you'll mount it manually in both OSes after install — easier than fighting with the installer's mount UI)
- OK.

### 5.5 Bootloader location

At the bottom of the screen: **"Device for boot loader installation"** → set to **`/dev/nvme0n1`** (the whole disk, NOT a partition like `nvme0n1p3`).

### 5.6 Continue and let it install

- Click **Install Now** → confirm the partition changes.
- Timezone: Zurich (or your locale).
- User account: pick a **different username and hostname** from your 24.04 install so you can distinguish them. E.g. hostname `ubuntu-2204`, username your choice.
- Wait for installation to finish.
- When prompted to reboot: click reboot, remove the USB when asked, press Enter.

---

## 6. First boot after install

The 22.04 installer wrote its GRUB into `/EFI/ubuntu/` on the EFI partition, replacing 24.04's GRUB shim there. That's expected.

You should see a GRUB menu with at least:

- Ubuntu (22.04 — default)
- Advanced options for Ubuntu

If **24.04 does not appear** in the menu, fix it from 22.04:

```bash
# Boot into 22.04, open terminal
sudo apt update
sudo apt install -y os-prober
# Allow os-prober (Debian/Ubuntu disable it by default for security)
echo 'GRUB_DISABLE_OS_PROBER=false' | sudo tee -a /etc/default/grub
sudo os-prober           # should list /dev/nvme0n1p2 as Ubuntu 24.04
sudo update-grub         # rebuilds menu with both entries
sudo reboot
```

After reboot, you should see both Ubuntus in the GRUB menu.

---

## 7. (Recommended) Make 24.04 the GRUB owner

The newer Ubuntu's GRUB handles newer kernels better. To put 24.04 in charge of the boot menu:

```bash
# Boot into 24.04 from the GRUB menu
sudo apt update
sudo apt install -y os-prober
echo 'GRUB_DISABLE_OS_PROBER=false' | sudo tee -a /etc/default/grub
sudo grub-install /dev/nvme0n1
sudo update-grub
```

Verify 22.04 shows up in the regenerated menu output. Reboot to confirm.

---

## 8. (Optional) Mount the shared data partition in both OSes

If you created the optional `/data` partition (Section 5.4):

```bash
# Find its UUID
sudo blkid | grep nvme0n1p4
# e.g. /dev/nvme0n1p4: UUID="abcd-1234-..." TYPE="ext4"

sudo mkdir /data
sudo nano /etc/fstab
# Add a line:
UUID=abcd-1234-...  /data  ext4  defaults,nofail  0  2

sudo mount -a
sudo chown $USER:$USER /data
```

Do the same on the other Ubuntu — both will then read/write `/data` natively.

**Do not share `/home` between the two Ubuntus** — config files from a newer GNOME can break the older one. Share data, not configs.

---

## 9. Verify everything

After reboot:

```bash
# Confirm UEFI entries
efibootmgr -v

# Confirm partition layout
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT

# Confirm GRUB sees both
sudo grep -E 'menuentry|submenu' /boot/grub/grub.cfg | head -20
```

Boot into each Ubuntu at least once and check:

- Network works
- Sound works
- GPU/display works
- `sudo apt update && sudo apt full-upgrade` runs clean

---

## 10. Recovery — if GRUB breaks

If a future kernel update or a Windows install (later) breaks GRUB and the machine won't boot:

1. Boot the 22.04 USB → "Try Ubuntu".
2. Install `boot-repair`:
   ```bash
   sudo add-apt-repository -y ppa:yannubuntu/boot-repair
   sudo apt update
   sudo apt install -y boot-repair
   boot-repair
   ```
3. Click **"Recommended repair"**. It will reinstall GRUB and detect both Ubuntus.

Alternative manual fix via chroot — only if `boot-repair` fails:

```bash
sudo mount /dev/nvme0n1p2 /mnt        # mount the Ubuntu you want as boot owner
sudo mount /dev/nvme0n1p1 /mnt/boot/efi
for d in dev proc sys run; do sudo mount --bind /$d /mnt/$d; done
sudo chroot /mnt
grub-install /dev/nvme0n1
update-grub
exit
```

---

## 11. Quick reference — what NOT to do

- ❌ Do NOT format `/dev/nvme0n1p1` (the EFI). It is shared.
- ❌ Do NOT pick "Erase disk and install Ubuntu" in the installer. It wipes 24.04.
- ❌ Do NOT pick "Install alongside" — the auto-detector gets confused by two Ubuntus.
- ❌ Do NOT install GRUB to a partition (`nvme0n1p3`). Always to the disk (`nvme0n1`).
- ❌ Do NOT share `/home` between the two Ubuntus.
- ❌ Do NOT interrupt power during the GParted resize.

---

## Appendix — Time estimate

| Step | Approx. time |
|---|---|
| Backup home (~100 GB to USB 3.0) | 30–60 min |
| Boot USB live session | 2 min |
| GParted shrink of 103 GB ext4 | 10–30 min |
| Ubuntu 22.04 install | 15–25 min |
| Post-install GRUB tweaks | 5–10 min |
| **Total** | **~1.5–2.5 hours** |
