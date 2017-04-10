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
      start = timeit.default_timer()
      sess.run(tens)
      end = timeit.default_timer()
      best_of_three = min(best_of_three, end - start)
    timings.append(best_of_three)
  return np.mean(timings)


def freeze_args(argument_list, sess):
  cheap_args = []
  for argument in argument_list:
    shape = sess.run(tf.shape(argument))
    cheap_args.append(tf.constant(np.random.rand(*shape)))
  return cheap_args


def optimize_einsum(struct, sess):
  subscripts = struct['subscripts']
  pos = subscripts.find('->')
  if pos != -1:
    output_str = subscripts[pos:]
    subscripts = subscripts[:pos]
  else:
    output_str = ''
  argument_strings = np.array(subscripts.split(','))

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
        raise ValueError('Calling different types of einsum from the same line of code '
                         'is not supported, %s sometimes calls einsum with argumens "%s"'
                         'and sometimes with "%s"' % (caller_str, cache[caller_str]['subscripts'],
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
    subscripts = cache[caller_str]['subscripts']
    arguments = cache[caller_str]['arguments']
    cache[caller_str]['cheap_args'] = []
    cur_timings = np.zeros(len(arguments))
    for i in range(len(arguments)):
      cheap_args = freeze_args(arguments[i], sess)
      cache[caller_str]['cheap_args'].append(cheap_args)
      curr_tens = original_einsum(subscripts, *cheap_args)
      cur_timings[i] = my_timeit(curr_tens, sess)
    cache[caller_str]['timings'] = cur_timings
  vanilla_einsum_runtime = [np.sum(cache[s]['timings']) for s in cache]
  print('Einsums constitue %0.1f %% of the running time of the whole function (%f s).' %
        (100 * np.sum(vanilla_einsum_runtime) / vanilla_whole_runtime, np.sum(vanilla_einsum_runtime)))

  worst_einsum_idx = np.argmax([np.max(cache[s]['timings']) for s in cache])
  worst_einsum = list(cache)[worst_einsum_idx]
  vanilla_wors_timings = cache[worst_einsum]['timings']
  print('The slowest einsum (on which we gonna focus) is located in %s and it '
        'constitues %0.1f %% of the running time of the whole function (%f s).' %
        (worst_einsum, 100 * np.sum(vanilla_wors_timings) / vanilla_whole_runtime, np.sum(vanilla_wors_timings)))

  if np.sum(vanilla_wors_timings) / vanilla_whole_runtime < 10:
    print('Nothing to improve, einsums are already too fast.')
    return

  print(cache)
  timings_table, orders = optimize_einsum(cache[worst_einsum], sess)
  print(vanilla_wors_timings, timings_table, np.sum(vanilla_wors_timings - timings_table, axis=1))
  absolute_savings = np.sum(vanilla_wors_timings - timings_table, axis=1)
  global_rel_savings = (absolute_savings) / float(vanilla_whole_runtime)
  best_order_idx = np.argmax(global_rel_savings)
  best_order = orders[best_order_idx]
  best_improovement = 100 * global_rel_savings[best_order_idx]
  if best_improovement >= 20:
    print('By changing the order of einsum in "%s" to %s you program will run %0.1f %% faster.' %
          (worst_einsum, best_order, best_improovement))
  else:
    print('Einsum improvements haven\'t found, good work!')
