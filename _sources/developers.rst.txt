=================
Developer's Guide
=================

Sending Your Work
=================

We accept pull requests made through GitHub. As is usual,
we request that the changes be rebased
on the branch they are to be integrated into.  We also request that
you pre-lint and test anything you send.

We'll try our best to attribute
your work to you, however, you need to release your work under
compatible license for us to be able to use it.

.. warning::

   We don't use `git-merge` command, and if your submission has merge
   commits, we'll have to remove them.  This means that in such case
   commit hashes will be different from those in your original
   submission.


Setting Up Development Environment
==================================


The Way We Do It
^^^^^^^^^^^^^^^^

This is a pip project.




Testing
=======

You may follow the README.dev.md , and apply manual testing as logical.


Style Guide for Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have linting! We use prospector.



Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

This project has CI setup that uses GitHub Actions
platform.



Style
^^^^^

When it comes to style, beyond linting we are trying
to conform, more or less, to the Google Python style
https://google.github.io/styleguide/pyguide.html
