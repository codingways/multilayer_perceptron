require "rubygems"
require "ai4r"
require 'RMagick'
include Magick

frankito_img = ImageList.new 'frankitobw.jpg'
frankito_input = frankito_img.first.get_pixels(0,0,frankito_img.first.columns,frankito_img.first.rows).collect { |pixel| [pixel.red,pixel.green,pixel.blue]}.flatten

juanpe_img = ImageList.new 'juanpebw.jpg'
juanpe_input = juanpe_img.first.get_pixels(0,0,juanpe_img.first.columns,juanpe_img.first.rows).collect { |pixel| [pixel.red,pixel.green,pixel.blue]}.flatten

tinchito_img = ImageList.new 'tinchitobw.jpg'
tinchito_input = tinchito_img.first.get_pixels(0,0,tinchito_img.first.columns,tinchito_img.first.rows).collect { |pixel| [pixel.red,pixel.green,pixel.blue]}.flatten

begin
  net = Marshal.load(File.read('net'))
  puts 'Cargando Net previa...'
rescue
  net = Ai4r::NeuralNetwork::Backpropagation.new([tinchito_input.size, 12,3])
  puts 'Inicializando Net por primera vez...'
end

puts "Entrenando la red..."
10.times do |i|
  puts (i + 1).to_s + " iteracion"
  net.train(juanpe_input, [1,0,0])   
  net.train(tinchito_input, [0,0,1]) 
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
juanpe_resultado = net.eval(juanpe_input)
tincho_resultado = net.eval(tinchito_input)

puts "#{juanpe_resultado.inspect} => #{result_label(juanpe_resultado)}"
puts "#{tincho_resultado.inspect} => #{result_label(tincho_resultado)}"

File.open('net','w') {|f| f.write(Marshal.dump(net))}
