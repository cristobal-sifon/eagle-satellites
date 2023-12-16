# eagle-satellites

Repository for the "The history and mass content of cluster galaxies in the EAGLE simulation" by Cristóbal Sifón and Jiaxin Han.

Only the ``hbt`` folder contains code and data used in the paper. All other folders are leftover code from original exploration of EAGLE's ``subfind`` catalogues and initial tests.

The ``hbt/hbtpy`` folder contains functionality to deal with the ``HBT+`` catalogues (contact Jiaxin Han to request access).

Figures in the paper were produced from the following:

* Figure 1: ``fit_relations.py``
* Figures 2, 3, 4, 5, 7, 8, and 12: ``make_plots.py``, which calls ``plot_relations.py``. In the latter, modify the functions ``wrap_relations_distance`` (Figs. 3 and 12), ``wrap_relations_hsmr`` (Figs. 2, 4, and 5), and ``wrap_relations_time`` (Figs. 7 and 8), specifying the ``xcols``, ``ycols``, ``bincols``, and ``selection_kwds`` appropriately.
* Figure 6: ``compare_times.py``
* Figure 9: ``shmr_history.py``
* Figures 10, 11, B.1, and B.2: ``preprocessing.py``
* Figure 13: ``massloss.py``
* Figure A.1: ``orphan.py``

Most of these require first having run ``store_subhalo_times.py`` to generate historical information.