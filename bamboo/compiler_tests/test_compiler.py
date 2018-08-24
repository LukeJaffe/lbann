import pytest
import os, re, subprocess

def build_script(cluster, dirname, compiler, debug):
    if debug:
        build = 'debug'
    else:
        build = 'release'
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_%s_output.txt' % (dirname, cluster, compiler, build)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_%s_error.txt' % (dirname, cluster, compiler, build)
    command = '%s/bamboo/compiler_tests/build_script.sh --compiler %s %s> %s 2> %s' % (dirname, compiler, debug, output_file_name, error_file_name)
    return_code = os.system(command)
    if return_code != 0:
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0

def test_compiler_clang4_release(cluster, dirname):
    #skeleton_clang4(cluster, dirname, False)
    if cluster in ['ray', 'catalyst']:
        build_script(cluster, dirname, 'clang', '')
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_clang4_debug(cluster, dirname):
    #skeleton_clang4(cluster, dirname, True)
    if cluster in ['ray', 'catalyst']:
        build_script(cluster, dirname, 'clang', '--debug')
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_gcc4_release(cluster, dirname):
    #skeleton_gcc4(cluster, dirname, False)
    build_script(cluster, dirname, 'gcc4', '')

def test_compiler_gcc4_debug(cluster, dirname):
    #skeleton_gcc4(cluster, dirname, True)
    build_script(cluster, dirname, 'gcc4', '--debug')

def test_compiler_gcc7_release(cluster, dirname):
    #skeleton_gcc7(cluster, dirname, False)
    if cluster == 'catalyst':
        build_script(cluster, dirname, 'gcc7', '')
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_gcc7_debug(cluster, dirname):
    #skeleton_gcc7(cluster, dirname, True)
    if cluster == 'catalyst':
        build_script(cluster, dirname, 'gcc7', '--debug')
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_intel18_release(cluster, dirname):
    #skeleton_intel18(cluster, dirname, False)
    if cluster == 'catalyst':
        build_script(cluster, dirname, 'intel', '')
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_intel18_debug(cluster, dirname):
    #skeleton_intel18(cluster, dirname, True)
    if cluster == 'catalyst':
        build_script(cluster, dirname, 'intel', '--debug')
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_clang4(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'quartz']:
        spack_skeleton(dir_name, 'clang@4.0.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'clang@4.0.0', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_gcc4(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'quartz', 'ray']:
        if cluster in ['catalyst','quartz']:
            mpi = 'mvapich2@2.2'
        elif cluster in  ['pascal', 'surface']:
            mpi = 'mvapich2@2.2+cuda'
        elif cluster == 'ray':
            mpi = 'spectrum-mpi@2018.04.27'
        else:
            raise Exception('Unsupported Cluster %s' % cluster)
        spack_skeleton(dir_name, 'gcc@4.9.3', mpi, debug, should_log)
        build_skeleton(dir_name, 'gcc@4.9.3', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_gcc7(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'quartz']:
        spack_skeleton(dir_name, 'gcc@7.1.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'gcc@7.1.0', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_intel18(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'quartz']:
        spack_skeleton(dir_name, 'intel@18.0.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'intel@18.0.0', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def spack_skeleton(dir_name, compiler, mpi_lib, debug, should_log):
    compiler_underscored = re.sub('[@\.]', '_', compiler)
    if debug:
        build_type = 'debug'
    else:
        build_type = 'rel'
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_spack_output.txt' % (dir_name, compiler_underscored, build_type)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_spack_error.txt' % (dir_name, compiler_underscored, build_type)
    os.chdir('%s/bamboo/compiler_tests/builds' % dir_name)
    debug_flag = ''
    if debug:
        debug_flag = ' -d'
    command = '%s/scripts/spack_recipes/build_lbann.sh -c %s -m %s%s > %s 2> %s' % (
        dir_name, compiler, mpi_lib, debug_flag, output_file_name, error_file_name)
    return_code = os.system(command)
    os.chdir('..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0

def build_skeleton(dir_name, compiler, debug, should_log):
    compiler_underscored = re.sub('[@\.]', '_', compiler)
    if debug:
        build_type = 'debug'
    else:
        build_type = 'rel'
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_build_output.txt' % (dir_name, compiler_underscored, build_type)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_build_error.txt' % (dir_name, compiler_underscored, build_type)
    compiler = compiler.replace('@', '-')
    #mpi_lib = mpi_lib.replace('@', '-')
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).strip())
    # For reference:
    # Commenting out for now. These additions to path name will likely return one day, so I am not removing them entirely
    # x86_64 <=> catalyst, pascal, quartz, surface
    # ppc64le <=> ray
    #architecture = subprocess.check_output('uname -m'.split()).strip()
    #if cluster == 'ray':
    #    architecture += '_gpu_cuda-9.2.64_cudnn-7.0'
    #elif cluster == 'pascal':
    #    architecture += '_gpu_cuda-9.1.85_cudnn-7.1'
    #elif cluster == 'surface':
    #    architecture += '_gpu'
    os.chdir('%s/bamboo/compiler_tests/builds/%s_%s_%s/build' % (dir_name, cluster, compiler, build_type))
    command = 'make -j all > %s 2> %s' % (output_file_name, error_file_name)
    return_code = os.system(command)
    os.chdir('../..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0
