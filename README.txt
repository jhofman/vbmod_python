introduction
---------
vbmod_python is python software for identifying modules in networks
(e.g. "community detection"), as described in "a bayesian approach to
network modularity."

this software is free for scientific use. please contact us if you
plan to use this software for commercial purposes. do not further
distribute without prior permission of the authors. if used in your
scientific work, please cite as:

Jake M. Hofman and Chris H. Wiggins, "A Bayesian Approach to Network
Modularity", Phys. Rev. Lett. 100, 258701 (2008) 


installation
----------
vbmod_python doesn't require any installation, but does require the
scipy package (http://www.scipy.org/Installing_SciPy). 

it currently defaults to a version that uses weave
(http://www.scipy.org/Weave) to include inline c code for better
performance in the vbmod_estep_inline. (a version of this code in
python is left as a comment, and can be used in lieu of the weave code
in the event of compilation problems, etc.)

weave will take some time to compile the c code on the first run, but
be much faster than the corresponding python code. this speedup arises
because there is a for loop over nodes in the code, which is extremely
slow when interpreted by python.


demo
----------
run 'vbmod.py' which generates an adjacency matrix for a modular
random network and runs variational bayes to infer the modular
structure from this adjacency matrix.

see documentation in files for further information.


license 
----------
copyright (c) 2007, 2008 jake hofman <jhofman@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
