#!/bin/sh

# print empty lines
for i in {1..5}; do echo; done

user=csifon
passwd=UWV21cdg

Mstar_min=1E7
Mass_min=1e8

## Aliases
# FoF
GroupCentreOfPotential_x=x
GroupCentreOfPotential_y=y
GroupCentreOfPotential_z=z
GroupMass=M
Group_M_Crit200=M200c
Group_M_Crit500=M500c
Group_M_Mean200=M200m
Group_R_Crit200=R200c
Group_R_Crit500=R500c
Group_R_Mean200=R200m
NumOfSubhalos=Nsub
# Subhalo
CentreOfMass_x=xm
CentreOfMass_y=ym
CentreOfMass_z=zm
CentreOfPotential_x=x
CentreOfPotential_y=y
CentreOfPotential_z=z
HalfMassRad_DM=rhalf_DM
HalfMassRad_Gas=rhalf_gas
HalfMassRad_Star=rhalf_star
HalfMassProjRad_DM=rphalf_DM
HalfMassProjRad_Gas=rphalf_gas
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
## End aliases


# ---

baseurl="http://galaxy-catalogue.dur.ac.uk:8080/Eagle?action=doQuery&SQL=select"

## actual code below

simulation=L0100N1504

for snapshot in {0..28}
do

#for physics in Ref DMONLY
#do
physics=Ref

    if [[ $physics == DMONLY ]]; then
        table=$physics..$simulation
    else
        table=$physics$simulation
    fi

output_dir=data/$simulation/$physics/snapshot$snapshot
# do we need to create the directory?
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi


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

from ${table}_FOF
where SnapNum = $snapshot
and Group_M_Mean200 > 1E11"

wget --http-user=$user --http-passwd=$passwd "$url" -O $output_dir/groups.txt


## ----
## Galaxies
## ----

columns_galaxy="GalaxyID, GroupID, \
TopLeafID, LastProgID, DescendantID, \
CentreOfMass_x as $CentreOfMass_x, \
CentreOfMass_y as $CentreOfMass_y, \
CentreOfMass_z as $CentreOfMass_z, \
CentreOfPotential_x as $CentreOfPotential_x, \
CentreOfPotential_y as $CentreOfPotential_y, \
CentreOfPotential_z as $CentreOfPotential_z, \
HalfMassRad_DM as $HalfMassRad_DM, \
HalfMassProjRad_DM as $HalfMassProjRad_DM, \
Mass as $Mass, \
MassType_DM as $MassType_DM, \
Velocity_x as $Velocity_x, \
Velocity_y as $Velocity_y, \
Velocity_z as $Velocity_z, \
Vmax as $Vmax, \
VmaxRadius as $VmaxRadius"

columns_hydro="HalfMassRad_Gas as $HalfMassRad_Gas, \
HalfMassRad_Star as $HalfMassRad_Star, \
HalfMassProjRad_Gas as $HalfMassProjRad_Gas, \
HalfMassProjRad_Star as $HalfMassProjRad_Star, \
MassType_Star as $MassType_Star, \
StarFormationRate as $StarFormationRate"

cuts_galaxy="
from ${table}_Subhalo
where SnapNum = $snapshot
and Mass >= $Mass_min"
if [ $physics != DMONLY ]; then
    cuts_galaxy="$cuts_galaxy and Spurious = 0 and MassType_Star > $Mstar_min"
fi

url="$baseurl $columns_galaxy"
if [ $physics != DMONLY ]; then
    url="$url, $columns_hydro"
fi
url="$url
$cuts_galaxy"


## ----
## Centrals
## ----

url_cent="$url
    and SubGroupNumber = 0"

wget --http-user=$user --http-passwd=$passwd "$url_cent" -O $output_dir/centrals.txt

## ----
## Satellites
## ----

url_sat="$url
    and SubGroupNumber > 0
    and Group_M_Mean200 > 1e13"

wget --http-user=$user --http-passwd=$passwd "$url_sat" -O $output_dir/satellites.txt


# closes the physics loop
#done

# closes the snapshot loop
done
