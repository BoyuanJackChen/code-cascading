package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestMaxFill(t *testing.T) {
  assert := assert.New(t)
  assert.Equal(6, MaxFill([][]int{{0,0,1,0}, {0,1,0,0}, {1,1,1,1}}, 1))
  assert.Equal(5, MaxFill([][]int{{0,0,1,1}, {0,0,0,0}, {1,1,1,1}, {0,1,1,1}}, 2))
  assert.Equal(0, MaxFill([][]int{{0,0,0}, {0,0,0}}, 5))
  assert.Equal(4, MaxFill([][]int{{1,1,1,1}, {1,1,1,1}}, 2))
  assert.Equal(2, MaxFill([][]int{{1,1,1,1}, {1,1,1,1}}, 9))
}
