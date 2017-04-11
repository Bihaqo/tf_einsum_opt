from inspect import getframeinfo, stack
import timeit
import itertools

import numpy as np
import tensorflow as tf


def my_timeit(tens, sess):
  timings = []
  for rep in range(20):
    best_of_three = np.inf
    for i in range(3):
      try:
        start = timeit.default_timer()
        sess.run(tens)
        end = timeit.default_timer()
        current_time = end - start
      except tf.errors.ResourceExhaustedError:
        # If the version we are profiling overflows memory, it probably doesn't
        # worth spending time on it.
        return np.inf
      best_of_three = min(best_of_three, current_time)
    timings.append(best_of_three)
  return np.mean(timings)


def freeze_args(argument_list, sess):
  cheap_args = []
  for argument in argument_list:
    shape = sess.run(tf.shape(argument))
    cheap_args.append(tf.constant(np.random.rand(*shape)))
  return cheap_args


def parse_subscripts(subscripts):
  pos = subscripts.find('->')
  if pos != -1:
    output_str = subscripts[pos:]
    subscripts = subscripts[:pos]
  else:
    output_str = ''
  argument_strings = np.array(subscripts.split(','))
  return argument_strings, output_str


def optimize_einsum(struct, sess):
  argument_strings, output_str = parse_subscripts(struct['subscripts'])
  cheap_args = np.array(struct['cheap_args'])
  num_args = len(cheap_args[0])
  orders = np.array(list(itertools.permutations(range(num_args))))
  num_orders = orders.shape[0]
  timings_table = np.zeros((num_orders, len(cheap_args)))
  for order_idx in range(num_orders):
    curr_order = orders[order_idx, :]
    curr_einsum_string = ','.join(argument_strings[curr_order])
    curr_einsum_string += output_str
    for i in range(len(cheap_args)):
      curr_tens = tf.einsum(curr_einsum_string, *cheap_args[i][curr_order])
      timings_table[order_idx, i] = my_timeit(curr_tens, sess)

  return timings_table, orders


def optimizer(f, sess, *args):
  cache = {}
  original_einsum = tf.einsum

  def my_einsum(subscripts, *args):
    caller = getframeinfo(stack()[1][0])
    caller_str = "%s:%d" % (caller.filename, caller.lineno)
    if caller_str in cache:
      if cache[caller_str]['subscripts'] != subscripts:
        raise ValueError('Calling different types of einsum from the same line '
                         'of code is not supported, %s sometimes calls einsum '
                         'with argumens "%s" and sometimes with "%s"' %
                         (caller_str, cache[caller_str]['subscripts'],
                          subscripts))
      cache[caller_str]['arguments'].append(args)
    else:
      cache[caller_str] = {'subscripts': subscripts, 'arguments': [args]}
    return original_einsum(subscripts, *args)
  tf.einsum = my_einsum
  f_out = f(*args)
  tf.einsum = original_einsum
  print('Found %d einsums.' % len(cache))
  vanilla_whole_runtime = my_timeit(f_out, sess)
  print('The running time of the whole function is %f s' % vanilla_whole_runtime)
  for caller_str in cache:
    curr_subscripts = cache[caller_str]['subscripts']
    arguments = cache[caller_str]['arguments']
    cache[caller_str]['cheap_args'] = []
    cur_timings = np.zeros(len(arguments))
    for i in range(len(arguments)):
      cheap_args = freeze_args(arguments[i], sess)
      cache[caller_str]['cheap_args'].append(cheap_args)
      curr_tens = original_einsum(curr_subscripts, *cheap_args)
      cur_timings[i] = my_timeit(curr_tens, sess)
    cache[caller_str]['timings'] = cur_timings
  vanilla_einsum_runtime = [np.sum(cache[s]['timings']) for s in cache.keys()]
  print('Einsums constitue %0.1f %% of the running time (%f s).' %
        (100 * np.sum(vanilla_einsum_runtime) / vanilla_whole_runtime,
         np.sum(vanilla_einsum_runtime)))

  slowest_to_fastest = np.argsort(vanilla_einsum_runtime)[::-1]
  rel_savings_combined = 0.0
  improved_orders = {}
  for idx in range(len(slowest_to_fastest)):
    caller_str = cache.keys()[slowest_to_fastest[idx]]
    vanilla_einsum_timings = cache[caller_str]['timings']
    rel_timing = np.sum(vanilla_einsum_timings) / vanilla_whole_runtime
    if rel_timing < 0.1:
      print('The rest of einsums are using < 10%% of the overall running time '
            'each, we will not gain much by optimizing them.')
      break

    print('Optimizing einsum in %s, it constitues %0.1f%% of the overall '
          'running time (%f s).' % (caller_str, 100 * rel_timing,
                                    np.sum(vanilla_einsum_timings)))

    timings_table, orders = optimize_einsum(cache[caller_str], sess)
    absolute_savings = np.sum(vanilla_einsum_timings - timings_table, axis=1)
    global_rel_savings = absolute_savings / float(vanilla_whole_runtime)
    best_order_idx = np.argmax(global_rel_savings)
    best_order = orders[best_order_idx]
    best_rel_improvement = global_rel_savings[best_order_idx]
    if best_rel_improvement >= 0.1:
      print('By changing the order of einsum in "%s" to %s you program will '
            'run %0.1f %% faster.' % (caller_str, best_order,
                                      100 * best_rel_improvement))
      rel_savings_combined += global_rel_savings[best_order_idx]
      improved_orders[caller_str] = best_order
    else:
      print('Einsum improvements haven\'t found, good work!')

  if rel_savings_combined > 0:
    print('The overall predicted savings from all the recommendations are %f%%' %
          (100 * rel_savings_combined))

  def my_optimizing_einsum(subscripts, *args):
    caller = getframeinfo(stack()[1][0])
    caller_str = "%s:%d" % (caller.filename, caller.lineno)
    if caller_str in improved_orders:
      order = improved_orders[caller_str]
      argument_strings, output_str = parse_subscripts(subscripts)
      subscripts = ','.join(argument_strings[order])
      subscripts += output_str
    return original_einsum(subscripts, *args)

  def optimized_func(*args):
    original_einsum = tf.einsum
    tf.einsum = my_optimizing_einsum
    res = f(*args)
    tf.einsum = original_einsum
    return res

  return improved_orders, optimized_func
