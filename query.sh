#!/bin/sh

snapshot=26
simulation=RefL0100N1504

Mstar_min=1E9

## Aliases
# FoF
GroupCentreOfPotential_x=x
GroupCentreOfPotential_y=y
GroupCentreOfPotential_z=z
GroupMass=M
Group_M_Crit200=M200c
Group_M_Crit500=M500c
Group_M_Mean200=M200a
Group_R_Crit200=R200c
Group_R_Crit500=R500c
Group_R_Mean200=R200m
NumOfSubhalos=Nsub
# Subhalo
CentreOfMass_x=x
CentreOfMass_y=y
CentreOfMass_z=z
CentreOfPotential_x=xgrav
CentreOfPotential_y=ygrav
CentreOfPotential_z=zgrav
HalfMassRad_DM=rhalf_DM
HalfMassRad_Star=rhalf_star
HalfMassProjRad_DM=rphalf_DM
HalfMassProjRad_Star=rphalf_star
Mass=M
MassType_DM=M_DM
MassType_Gas=Mgas
MassType_Star=Mstar
StarFormationRate=sfr
Velocity_x=vx
Velocity_y=vy
Velocity_z=vz
Vmax=vmax
VmaxRadius=r_vmax

output_dir=data/$simulation/snapshot$snapshot
# do we need to create the directory?
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

baseurl="http://galaxy-catalogue.dur.ac.uk:8080/Eagle?action=doQuery&SQL=select"

## ----
## Massive groups
## ----

url="$baseurl \
GroupID, \
GroupCentreOfPotential_x as $GroupCentreOfPotential_x, \
GroupCentreOfPotential_y as $GroupCentreOfPotential_y, \
GroupCentreOfPotential_z as $GroupCentreOfPotential_z, \
GroupMass as $GroupMass, \
Group_M_Crit200 as $Group_M_Crit200, \
Group_M_Crit500 as $Group_M_Crit500, \
Group_M_Mean200 as $Group_M_Mean200, \
Group_R_Crit200 as $Group_R_Crit200, \
Group_R_Crit500 as $Group_R_Crit500, \
Group_R_Mean200 as $Group_R_Mean200, \
NumOfSubhalos as $NumOfSubhalos

from ${simulation}_FOF
where SnapNum = $snapshot
and Group_M_Mean200 > 1E12"

wget --http-user=csifon --http-passwd=UWV21cdg "$url" -O $output_dir/groups.txt


## ----
## Galaxies
## ----

columns_galaxy="GalaxyID, GroupID, Redshift as z, \
CentreOfMass_x as $CentreOfMass_x, \
CentreOfMass_y as $CentreOfMass_y, \
CentreOfMass_z as $CentreOfMass_z, \
CentreOfPotential_x as $CentreOfPotential_x, \
CentreOfPotential_y as $CentreOfPotential_y, \
CentreOfPotential_z as $CentreOfPotential_z, \
HalfMassRad_DM as $HalfMassRad_DM, \
HalfMassRad_Star as $HalfMassRad_Star, \
HalfMassProjRad_DM as $HalfMassProjRad_DM, \
HalfMassProjRad_Star as $HalfMassProjRad_Star, \
Mass as $Mass, \
MassType_DM as $MassType_DM, \
MassType_Star as $MassType_Star, \
StarFormationRate as $StarFormationRate, \
Velocity_x as $Velocity_x, \
Velocity_y as $Velocity_y, \
Velocity_z as $Velocity_z, \
Vmax as $Vmax, \
VmaxRadius as $VmaxRadius"

cuts_galaxy="
from ${simulation}_Subhalo
where SnapNum = $snapshot
and Spurious = 0
and Stars_Mass > $Mstar_min"


## ----
## Centrals
## ----

url="$baseurl $columns_galaxy
$cuts_galaxy
and SubGroupNumber = 0"

wget --http-user=csifon --http-passwd=UWV21cdg "$url" -O $output_dir/centrals.txt

## ----
## Satellites
## ----

url="$baseurl $columns_galaxy
$cuts_galaxy
and SubGroupNumber > 0"

wget --http-user=csifon --http-passwd=UWV21cdg "$url" -O $output_dir/satellites.txt


## ----
## Merger history
## ----

url="$baseurl GalaxyID, LastProgID, TopLeafID, DescendantID
$cuts_galaxy"

wget --http-user=csifon --http-passwd=UWV21cdg "$url" -O $output_dir/history.txt
