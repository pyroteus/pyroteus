import firedrake
import ufl


__all__ = ["transfer_form"]


def transfer_form(F, newmesh, transfer=firedrake.prolong, replace_map={}):
    """
    Given a form defined on some mesh, generate a new form with all
    the same components, but transferred onto a different mesh.

    :arg F: the form to be transferred
    :arg newmesh: the mesh to transfer the form to
    :kwarg transfer: the transfer operator to use
    :kwarg replace_map: user-provided replace map
    """
    f = 0

    # We replace at least the coordinate map
    replace_map = {
        ufl.SpatialCoordinate(F.ufl_domain()): ufl.SpatialCoordinate(newmesh)
    }

    # Test and trial functions are also replaced
    if len(F.arguments()) > 0:
        Vold = F.arguments()[0].function_space()
        Vnew = firedrake.FunctionSpace(newmesh, Vold.ufl_element())
        replace_map[firedrake.TestFunction(Vold)] = firedrake.TestFunction(Vnew)
        replace_map[firedrake.TrialFunction(Vold)] = firedrake.TrialFunction(Vnew)

    # As well as any spatially varying coefficients
    for c in F.coefficients():
        if isinstance(c, firedrake.Function) and c not in replace_map:
            replace_map[c] = firedrake.Function(firedrake.FunctionSpace(newmesh, c.ufl_element()))
            transfer(c, replace_map[c])

    # The form is reconstructed by cell type
    for cell_type, dX in zip(('cell', 'exterior_facet', 'interior_facet'), (ufl.dx, ufl.ds, ufl.dS)):
        for integral in F.integrals_by_type(cell_type):
            differential = dX(integral.subdomain_id(), domain=newmesh)
            f += ufl.replace(integral.integrand(), replace_map)*differential

    return f
