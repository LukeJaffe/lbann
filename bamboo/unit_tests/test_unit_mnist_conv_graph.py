import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os

def skeleton_mnist_conv_graph(cluster, executables, dir_name, compiler_name):
    if compiler_name not in executables:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)
    output_file_name = '%s/bamboo/unit_tests/output/mnist_conv_graph_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/mnist_conv_graph_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name], num_nodes=1, num_processes=1,
        dir_name=dir_name, data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests', model_name='mnist_conv_graph',
        optimizer_name='adam',
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code = os.system(command)
    assert return_code == 0

def test_unit_mnist_conv_graph_clang4(cluster, exes, dirname):
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'clang4')

def test_unit_mnist_conv_graph_gcc4(cluster, exes, dirname):
    if cluster in ['surface']:
        pytest.skip('FIXME')
        # Surface Errors:
        # assert 35584 == 0
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'gcc4')

def test_unit_mnist_conv_graph_gcc7(cluster, exes, dirname):
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'gcc7')

def test_unit_mnist_conv_graph_intel18(cluster, exes, dirname):
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'intel18')

# Run with python -m pytest -s test_unit_conv_graph.py -k 'test_unit_mnist_conv_graph_exe' --exe=<executable>
def test_unit_mnist_conv_graph_exe(cluster, dirname, exe):
    if exe == None:
        pytest.skip('Non-local testing')
    exes = {'exe' : exe}
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'exe')
