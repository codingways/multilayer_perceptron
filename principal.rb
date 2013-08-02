require "rubygems"
require "ai4r"
require 'RMagick'
include Magick

def obtener_entradas(nombre_imagen)
  img = Image.read nombre_imagen
  img.first.get_pixels(0, 0, img.first.columns, img.first.rows).collect { |pixel| [pixel.red, pixel.green, pixel.blue] }.flatten
end

# Datos de entrenamiento
#input_franco = obtener_entradas 'entrenamiento/francobw.jpg'
input_juanpe1 = obtener_entradas 'entrenamiento/juanpe1.jpg'
input_juanpe2 = obtener_entradas 'entrenamiento/juanpe2.jpg'
input_juanpe3 = obtener_entradas 'entrenamiento/juanpe3.jpg'
input_tincho1 = obtener_entradas 'entrenamiento/tinchito1.jpg'
input_tincho2 = obtener_entradas 'entrenamiento/tinchito2.jpg'
input_tincho3 = obtener_entradas 'entrenamiento/tinchito3.jpg'

# Datos de prueba
#input2_juanpe = obtener_entradas 'prueba/juanpebw.jpg'
#input3_juanpe = obtener_entradas 'prueba/juanpe2.jpg'
#input2_tincho = obtener_entradas 'prueba/tinchobw.jpg'

begin
  net = Marshal.load(File.read('net'))
  puts 'Cargando Net previa...'
rescue
  net = Ai4r::NeuralNetwork::Backpropagation.new([input_juanpe.size, 12,3])
  puts 'Inicializando Net por primera vez...'
end

puts "Entrenando la red..."
100.times do |i|
  puts (i + 1).to_s + " iteracion"
  net.train(input_juanpe1, [1,0,0])   
  net.train(input_juanpe2, [1,0,0])   
  net.train(input_juanpe3, [1,0,0])   
  net.train(input_tincho1, [0,0,1]) 
  net.train(input_tincho2, [0,0,1])
  net.train(input_tincho3, [0,0,1])
end

def result_label(result)
  if result[0] > result[1] && result[0] > result[2]
    "JUANPE"
  elsif result[2] > result[1] && result[2] > result[0]
    "TINCHO"
  else
    "OTRO"
  end
end

puts "Resultados de entrenamiento:"
juanpe_resultado = net.eval(input_juanpe1)
tincho_resultado = net.eval(input_tincho1)

puts "#{juanpe_resultado.inspect} => #{result_label(juanpe_resultado)}"
puts "#{tincho_resultado.inspect} => #{result_label(tincho_resultado)}"

#puts "Resultados de otras pruebas de fotos:"
#juanpe_resultado2 = net.eval(input2_juanpe)
#juanpe_resultado3 = net.eval(input3_juanpe)
#tincho_resultado2 = net.eval(input2_tincho)
#puts "#{juanpe_resultado2.inspect} => #{result_label(juanpe_resultado2)}"
#puts "#{juanpe_resultado3.inspect} => #{result_label(juanpe_resultado3)}"
#puts "#{tincho_resultado2.inspect} => #{result_label(tincho_resultado2)}"

File.open('net','w') {|f| f.write(Marshal.dump(net))}
