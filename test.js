const { WebR } = require('webr'); // first = Ridham, last = Tarpara

WebR.evalR('rnorm(10,5,1)').then(async (result) => {
  let output = await result.toArray();

}).catch((err) => {
});

