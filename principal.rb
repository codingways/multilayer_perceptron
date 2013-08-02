require "rubygems"
require "ai4r"
require 'RMagick'
include Magick

def obtener_entradas(nombre_imagen)
  img = ImageList.new nombre_imagen
  img.first.get_pixels(0, 0, img.first.columns, img.first.rows).collect { |pixel| [pixel.red, pixel.green, pixel.blue] }.flatten
end

# Datos de entrenamiento
input_franco = obtener_entradas 'entrenamiento/francobw.jpg'
input_juanpe = obtener_entradas 'entrenamiento/juanpebw.jpg'
input_tincho = obtener_entradas 'entrenamiento/tinchobw.jpg'

# Datos de prueba
input2_juanpe = obtener_entradas 'prueba/juanpebw.jpg'
input2_tincho = obtener_entradas 'prueba/tinchobw.jpg'

begin
  net = Marshal.load(File.read('net'))
  puts 'Cargando Net previa...'
rescue
  net = Ai4r::NeuralNetwork::Backpropagation.new([input_juanpe.size, 12,3])
  puts 'Inicializando Net por primera vez...'
end

puts "Entrenando la red..."
1.times do |i|
  puts (i + 1).to_s + " iteracion"
  net.train(input_juanpe, [1,0,0])   
  net.train(input_tincho, [0,0,1]) 
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
juanpe_resultado = net.eval(input_juanpe)
tincho_resultado = net.eval(input_tincho)

puts "#{juanpe_resultado.inspect} => #{result_label(juanpe_resultado)}"
puts "#{tincho_resultado.inspect} => #{result_label(tincho_resultado)}"

puts "Resultados de otras pruebas de fotos:"
juanpe_resultado2 = net.eval(input2_juanpe)
tincho_resultado2 = net.eval(input2_tincho)
puts "#{juanpe_resultado2.inspect} => #{result_label(juanpe_resultado2)}"
puts "#{tincho_resultado2.inspect} => #{result_label(tincho_resultado2)}"

File.open('net','w') {|f| f.write(Marshal.dump(net))}
