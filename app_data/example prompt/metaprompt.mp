// Prompts can be commented. All lines prefixed by '//' will be stripped before
// the meta prompt is parsed.

// The root has 11 required fields. If you provide multiple options for cfg and steps,
// then one will be randomly chosen. The mutation frequency determines the chance that
// a particular prompt element will have its strength randomly increased or decreased.
//
// 'adds' declares which nodes will be added as children of this node. By
// default, every single node mentioned in 'adds' will be added to the prompt.
root: 
	set name: example prompt
	set author: David Vorick
	checkpoint: sd-1.5
	vae: sd-1.5
	width: 512
	height: 768
	upscale: 1.5
	sampler: DPM++ 2M Karras
	cfg: 7, 9, 11
	steps: 40, 64, 80
	mutation frequency: 0.25
adds:
	location
	theme
	subject
	sfw-safety

// 'prompts' states a list of prompts that can be selected. By default, exactly
// one prompt is selected from a list.
node: location
prompts:
	on the beach
	at a swimming pool
	by a water fountain

// Prompts can declare a floating point number. If a number is declared, a
// random number of prompts between 0 and every prompt will be selected. The
// number establishes how many prompts will be selected on average. With a
// value of 10, there is a 10% chance that zero prompts are selected.
//
// If you specify 'all' instead of a floating point, then every single prompt
// will be selected every time.
//
// An individual prompt can also use a floating point number. Saying '2.5'
// here means that the prompt is 2.5x as likely to be selected as all the
// other prompts. If a prompt is randomly selected multiple times, the
// extra selections will be ignored.
node: theme
prompts: 10.0
	2.5: masterpiece
	high quality
	high detail
	!ugly
	!low quality
	!author signature
	!jpeg artifacts

// If you only wish to select one node from a set of 'adds', you can specify
// 'one'. Here we ensure add either a man or a woman to our prompt. By
// separating men and women into different nodes, we can use gender specific
// prompting when we randomize. Splitting nodes like this gives us a great
// degree of control over the sets of images that is produced by our metaprompt.
node: subject
adds: one
	woman
	man

node: woman
adds:
	woman clothing
	woman accessories
prompts:
	an attractive woman
	an athletic woman
	an elegant woman

node: woman clothing
prompts:
	swimsuit
	sundress
	tank top and shorts

// some of the women produced by our metaprompt will have no accessories,
// others may have every accessory. On average though, they will have
// roughly two accessories each.
node: woman accessories
prompts: 2
	ear rings
	diamond ring
	eyeliner
	bracelet
	anklet

node: man
adds:
	man clothing
	man accessories
prompts:
	an attractive man
	a muscular man
	a distinguished man

node: man clothing
prompts:
	swim trunks
	tropical shirt and shorts

// By using a prompt frequency of less than one, we ensure that most men will
// not have any accessories. Some will have one accessory, and a few will have
// both.
node: man accessories
prompts: 0.3
	watch
	wedding band
	bandana

// We want to avoid unsafe content in our example, this does not give any
// guarantees but it helps a lot.
node: sfw-safety
prompts: all
	!nude
	!sexy
