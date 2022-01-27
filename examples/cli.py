import click
from click_repl import repl
import sys
# sys.path.append("../")
# import networkt
# import matplotlib.pyplot as plt
from cli_utils import *
from prompt_toolkit.history import FileHistory


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        myrepl()


@cli.group()
def graph():
    pass


@cli.group()
def statistics():
    pass


def myrepl():
    prompt_kwargs = {
        'history': FileHistory('./history'),
    }
    repl(click.get_current_context(), prompt_kwargs=prompt_kwargs)


if __name__ == "__main__":
    from clique_counter import *
    from utils import *
    from abcselection import runABC, ABCModelChoice
    import networkt

    add_to_click_cli(graph, networkt.padme_graph)
    add_to_click_cli(graph, generate_based_on_best_paras)
    add_to_click_cli(statistics, count_cliques)

    add_to_click_cli(statistics, distance_edge_best)
    add_to_click_cli(statistics, distance_edge)
    add_to_click_cli(statistics, mean_nbr_degree_combined)
    add_to_click_cli(statistics, mean_nbr_degree)
    add_to_click_cli(statistics, mean_clique_size_combined)
    add_to_click_cli(statistics, mean_clique_size)
    add_to_click_cli(statistics, num_cliques_combined)
    add_to_click_cli(statistics, num_cliques)
    add_to_click_cli(statistics, scaling_laws_combined)
    add_to_click_cli(statistics, scaling_laws)

    add_to_click_cli(statistics, runABC)
    add_to_click_cli(statistics, ABCModelChoice.apply_custom_dist)
    add_to_click_cli(statistics, ABCModelChoice.parse_results)

    cli()
