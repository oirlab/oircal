Installation
============

Python Environment
------------------

It is highly recommended users create a new Python environment with either `venv <https://docs.python.org/3/library/venv.html>`_, `Anaconda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_, or `uv <https://docs.astral.sh/uv/pip/environments/>`_. ``oircal`` requires Python 3.13 or newer.

To create a **uv** environment (in this example, called **oircal**)::

    uv venv oircal --python 3.13

Once you have created your **uv** environment, activate it::

    source oircal/bin/activate


Install from GitHub
-------------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/oirlab/oircal

2. Install live version with pip:

.. code-block:: bash

   cd oircal
   pip install -e .

The ``-e`` (editable) flag uses the specified package directory as the "live" source code, and does not copy it to the usual site-packages directory.

Installing builds the Cython extension used by the ``cython_likely`` ramp fitting algorithm, so a C compiler must be available.


Run the unit tests (optional)
-----------------------------

First install the testing dependencies. Quotes are needed for some shells:

.. code-block:: bash

  pip install -e ".[test]"

Run the tests with:

.. code-block:: bash

  pytest


Build the documentation (optional)
----------------------------------

.. code-block:: bash

  pip install -e ".[docs]"
  cd docs
  make html

The HTML pages are written to ``docs/build/html``.
